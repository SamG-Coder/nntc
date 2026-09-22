// nntc_view_vk: the asset on a real Vulkan sampler. STAGE 3 of docs/VULKAN_VIEWER_PLAN.md - parity with the Direct3D
// 11 viewer. It reads the descriptor and the .dds files it names, uploads every level of every one of them, builds the
// eight samplers, compiles bin/view.vert and bin/view.frag at runtime and draws the decoded material; and it has
// every key that viewer has with the same letters and the same meaning, its four-line overlay, its load-time BC4 / BC5
// pack of an uncompressed level 0 (key 4, through the same shared/bc_pack.h), and every one of its flags.
//
// Usage: nntc_view_vk <PREFIX_nntc.json> [flags]
// Controls: arrows move, W/S zoom, A/D yaw, Q/E pitch, Shift slow, C cube/quad, P/B/T point/bilinear/trilinear,
//           X anisotropy on / off (trilinear only), N next output texture, M mips on / off (the sampler's maxLod 0),
//           L level 1's LOD shift on / off (the 1:1 mip rule, carried by the gradients), V renormalise the shown
//           triple as a tangent-space normal, 1 show level 0's texture as stored, 2 show level 1's, 4 level 0 from
//           the BC4 / BC5 pack made at load, R reload both shaders from disk, Space reset, Esc quit, F1 a help
//           page of every key (any key closes it).
//
// The program this one is a sibling of is viewer/main.cpp, the Direct3D 11 viewer, and the reason a second viewer
// exists at all is that the format's claim is about THE hardware sampling operator rather than one vendor's: a second
// graphics api on the same asset is the cheapest evidence there is. The window rules, the camera, the geometry, the
// flag spellings, the asset's refusals and the .bmp writer are therefore lifted from that viewer rather than
// reinvented, so that the two stay comparable by construction - and the two --shot frames of one asset are compared
// with tools/frame_diff.py, which is what makes "the same picture" a number rather than an opinion.
//
// Vulkan 1.3 core and nothing else: dynamic rendering (no VkRenderPass, no VkFramebuffer) and synchronisation2 (one
// VkImageMemoryBarrier2 per layout change) are what make this file readable top to bottom, and both are core since
// January 2022. The loader is linked normally as vulkan-1 through CMake's FindVulkan - no volk, nothing vendored - and
// the instance asks for VK_KHR_surface plus the platform's surface extension, with the validation layer added in a
// Debug build when the machine has one.
//
// --shot FILE.bmp creates NO WINDOW AND NO SURFACE: an offscreen image, the same recording, a copy back through a
// staging buffer and the same 24-bit bottom-up .bmp the Direct3D viewer writes. That is what lets the release gate
// check a frame over a remote session, in a service, and on a Linux box with no desktop - and, like the other viewer's
// shot, it reads no keyboard state at all, so its bytes depend on the command line and on nothing else.
//
// Dependencies: the C runtime, the Vulkan loader, shaderc (the SDK's static one on Windows, the distribution's
// shared one on Linux), and the window system: Win32 on Windows, GLFW 3 on Linux.

// vulkan.h pulls windows.h in for the Win32 surface, so the two macros that keep that header small and keep it from
// defining min and max as macros have to be set before it, not beside the platform block below.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#include <vulkan/vulkan.h>
#include <shaderc/shaderc.h>

#include <cstdio>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>   // readlink, which is how exe_dir finds the running image on Linux (/proc/self/exe)
#endif

#include "nntc_json.h"   // the JSON reader, in shared/ and shared with the other viewer and the encoder
#include "dds.h"         // the DX10 .dds parsing, in shared/ because both viewers read the same files through it
#include "bc_pack.h"     // the BC4 / BC5 pack of an uncompressed level 0 (key 4), the encoder's own and the other viewer's

// ---------------------------------------------------------------------------
// WHICH VULKAN HEADERS THIS FILE NEEDS, and the one optional extension it can be built without.
//
// The floor is header 204. Vulkan 1.3 core is what the whole file is written on - dynamic rendering and
// synchronisation2 are used in every frame - and 1.3.204 is the first header that has them, so anything older
// cannot compile this and the #error below says so by name rather than producing a page of missing types.
//
// The ceiling is the cooperative-vector path, and it is OPTIONAL at compile time for the same reason it is optional
// at run time. VK_NV_cooperative_vector first appears in Vulkan-Headers 1.4.307, and a distribution's stock headers
// are routinely older than that (Ubuntu 22.04 ships 204, Debian 12 ships 239, Ubuntu 24.04 ships 275), so a build
// against them would fail on types, enums, PFNs and the extension-name macro that simply are not there. The header
// defines VK_NV_cooperative_vector itself when it carries the extension, so that macro - the header's own, not one
// invented here - is what decides. Compiled out, the viewer behaves exactly as it does on a device that lacks the
// extension: the query refuses in one line, --coopvec 1 is an error, and the plain GLSL decode, which is the
// product and runs everywhere, is what draws. Compiled in, every line below is the same line it was before the
// guard existed, which is what the gate's byte-identical frame checks are for.
// ---------------------------------------------------------------------------
#if VK_HEADER_VERSION < 204
#error "nntc_view_vk needs Vulkan 1.3 headers (VK_HEADER_VERSION 204 or newer): install libvulkan-dev on Debian or Ubuntu, vulkan-headers on Fedora or Arch, or the LunarG SDK"
#endif

#if defined(VK_NV_cooperative_vector)
#define NNTC_HAVE_COOPVEC 1
#else
#define NNTC_HAVE_COOPVEC 0
#endif

// VK_HEADER_VERSION as text, for the refusal the compiled-out build prints: a reader of that line needs the number
// their headers actually are, and the preprocessor is the only thing that knows it.
#define NNTC_STRINGIFY_(x) #x
#define NNTC_STRINGIFY(x) NNTC_STRINGIFY_(x)

#include "vk_check.h"   // VK_CHECK and vk_result_name: the one error path for the api, shared with the platform files

// ---------------------------------------------------------------------------
// The overlay's 8x8 font, ASCII 32-127, and it is THE OTHER VIEWER'S ARRAY: the same glyphs the Direct3D strip is
// drawn with (the author's own g_debug_font8x8_basic, which viewer/main.cpp carries), so that the two debug strips
// read identically rather than merely similarly. Nothing third-party is pulled in for it.
//
// Bit order: pixel (x, y) is set if (glyph[y] >> x) & 1 - the least significant bit is the leftmost column and y = 0
// is the top row.
// ---------------------------------------------------------------------------
static const uint8_t g_font8x8[96][8] = {
 { 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, { 0x18,0x3C,0x3C,0x18,0x18,0x00,0x18,0x00}, { 0x36,0x36,0x00,0x00,0x00,0x00,0x00,0x00}, { 0x36,0x36,0x7F,0x36,0x7F,0x36,0x36,0x00},
 { 0x0C,0x3E,0x03,0x1E,0x30,0x1F,0x0C,0x00}, { 0x00,0x63,0x33,0x18,0x0C,0x66,0x63,0x00}, { 0x1C,0x36,0x1C,0x6E,0x3B,0x33,0x6E,0x00}, { 0x06,0x06,0x03,0x00,0x00,0x00,0x00,0x00},
 { 0x18,0x0C,0x06,0x06,0x06,0x0C,0x18,0x00}, { 0x06,0x0C,0x18,0x18,0x18,0x0C,0x06,0x00}, { 0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00}, { 0x00,0x0C,0x0C,0x3F,0x0C,0x0C,0x00,0x00},
 { 0x00,0x00,0x00,0x00,0x00,0x0C,0x0C,0x06}, { 0x00,0x00,0x00,0x3F,0x00,0x00,0x00,0x00}, { 0x00,0x00,0x00,0x00,0x00,0x0C,0x0C,0x00}, { 0x60,0x30,0x18,0x0C,0x06,0x03,0x01,0x00},
 { 0x3E,0x63,0x73,0x7B,0x6F,0x67,0x3E,0x00}, { 0x0C,0x0E,0x0C,0x0C,0x0C,0x0C,0x3F,0x00}, { 0x1E,0x33,0x30,0x1C,0x06,0x33,0x3F,0x00}, { 0x1E,0x33,0x30,0x1C,0x30,0x33,0x1E,0x00},
 { 0x38,0x3C,0x36,0x33,0x7F,0x30,0x78,0x00}, { 0x3F,0x03,0x1F,0x30,0x30,0x33,0x1E,0x00}, { 0x1C,0x06,0x03,0x1F,0x33,0x33,0x1E,0x00}, { 0x3F,0x33,0x30,0x18,0x0C,0x0C,0x0C,0x00},
 { 0x1E,0x33,0x33,0x1E,0x33,0x33,0x1E,0x00}, { 0x1E,0x33,0x33,0x3E,0x30,0x18,0x0E,0x00}, { 0x00,0x0C,0x0C,0x00,0x00,0x0C,0x0C,0x00}, { 0x00,0x0C,0x0C,0x00,0x00,0x0C,0x0C,0x06},
 { 0x18,0x0C,0x06,0x03,0x06,0x0C,0x18,0x00}, { 0x00,0x00,0x3F,0x00,0x00,0x3F,0x00,0x00}, { 0x06,0x0C,0x18,0x30,0x18,0x0C,0x06,0x00}, { 0x1E,0x33,0x30,0x18,0x0C,0x00,0x0C,0x00},
 { 0x3E,0x63,0x7B,0x7B,0x7B,0x03,0x1E,0x00}, { 0x0C,0x1E,0x33,0x33,0x3F,0x33,0x33,0x00}, { 0x3F,0x66,0x66,0x3E,0x66,0x66,0x3F,0x00}, { 0x3C,0x66,0x03,0x03,0x03,0x66,0x3C,0x00},
 { 0x1F,0x36,0x66,0x66,0x66,0x36,0x1F,0x00}, { 0x7F,0x46,0x16,0x1E,0x16,0x46,0x7F,0x00}, { 0x7F,0x46,0x16,0x1E,0x16,0x06,0x0F,0x00}, { 0x3C,0x66,0x03,0x03,0x73,0x66,0x7C,0x00},
 { 0x33,0x33,0x33,0x3F,0x33,0x33,0x33,0x00}, { 0x1E,0x0C,0x0C,0x0C,0x0C,0x0C,0x1E,0x00}, { 0x78,0x30,0x30,0x30,0x33,0x33,0x1E,0x00}, { 0x67,0x66,0x36,0x1E,0x36,0x66,0x67,0x00},
 { 0x0F,0x06,0x06,0x06,0x46,0x66,0x7F,0x00}, { 0x63,0x77,0x7F,0x7F,0x6B,0x63,0x63,0x00}, { 0x63,0x67,0x6F,0x7B,0x73,0x63,0x63,0x00}, { 0x1C,0x36,0x63,0x63,0x63,0x36,0x1C,0x00},
 { 0x3F,0x66,0x66,0x3E,0x06,0x06,0x0F,0x00}, { 0x1E,0x33,0x33,0x33,0x3B,0x1E,0x38,0x00}, { 0x3F,0x66,0x66,0x3E,0x36,0x66,0x67,0x00}, { 0x1E,0x33,0x07,0x0E,0x38,0x33,0x1E,0x00},
 { 0x3F,0x2D,0x0C,0x0C,0x0C,0x0C,0x1E,0x00}, { 0x33,0x33,0x33,0x33,0x33,0x33,0x3F,0x00}, { 0x33,0x33,0x33,0x33,0x33,0x1E,0x0C,0x00}, { 0x63,0x63,0x63,0x6B,0x7F,0x77,0x63,0x00},
 { 0x63,0x63,0x36,0x1C,0x1C,0x36,0x63,0x00}, { 0x33,0x33,0x33,0x1E,0x0C,0x0C,0x1E,0x00}, { 0x7F,0x63,0x31,0x18,0x4C,0x66,0x7F,0x00}, { 0x1E,0x06,0x06,0x06,0x06,0x06,0x1E,0x00},
 { 0x03,0x06,0x0C,0x18,0x30,0x60,0x40,0x00}, { 0x1E,0x18,0x18,0x18,0x18,0x18,0x1E,0x00}, { 0x08,0x1C,0x36,0x63,0x00,0x00,0x00,0x00}, { 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF},
 { 0x0C,0x0C,0x18,0x00,0x00,0x00,0x00,0x00}, { 0x00,0x00,0x1E,0x30,0x3E,0x33,0x6E,0x00}, { 0x07,0x06,0x06,0x3E,0x66,0x66,0x3B,0x00}, { 0x00,0x00,0x1E,0x33,0x03,0x33,0x1E,0x00},
 { 0x38,0x30,0x30,0x3E,0x33,0x33,0x6E,0x00}, { 0x00,0x00,0x1E,0x33,0x3F,0x03,0x1E,0x00}, { 0x1C,0x36,0x06,0x0F,0x06,0x06,0x0F,0x00}, { 0x00,0x00,0x6E,0x33,0x33,0x3E,0x30,0x1F},
 { 0x07,0x06,0x36,0x6E,0x66,0x66,0x67,0x00}, { 0x0C,0x00,0x0E,0x0C,0x0C,0x0C,0x1E,0x00}, { 0x30,0x00,0x30,0x30,0x30,0x33,0x33,0x1E}, { 0x07,0x06,0x66,0x36,0x1E,0x36,0x67,0x00},
 { 0x0E,0x0C,0x0C,0x0C,0x0C,0x0C,0x1E,0x00}, { 0x00,0x00,0x33,0x7F,0x7F,0x6B,0x63,0x00}, { 0x00,0x00,0x1F,0x33,0x33,0x33,0x33,0x00}, { 0x00,0x00,0x1E,0x33,0x33,0x33,0x1E,0x00},
 { 0x00,0x00,0x3B,0x66,0x66,0x3E,0x06,0x0F}, { 0x00,0x00,0x6E,0x33,0x33,0x3E,0x30,0x78}, { 0x00,0x00,0x3B,0x6E,0x66,0x06,0x0F,0x00}, { 0x00,0x00,0x3E,0x03,0x1E,0x30,0x1F,0x00},
 { 0x08,0x0C,0x3E,0x0C,0x0C,0x2C,0x18,0x00}, { 0x00,0x00,0x33,0x33,0x33,0x33,0x6E,0x00}, { 0x00,0x00,0x33,0x33,0x33,0x1E,0x0C,0x00}, { 0x00,0x00,0x63,0x6B,0x7F,0x7F,0x36,0x00},
 { 0x00,0x00,0x63,0x36,0x1C,0x36,0x63,0x00}, { 0x00,0x00,0x33,0x33,0x33,0x3E,0x30,0x1F}, { 0x00,0x00,0x3F,0x19,0x0C,0x26,0x3F,0x00}, { 0x38,0x0C,0x0C,0x07,0x0C,0x0C,0x38,0x00},
 { 0x18,0x18,0x18,0x00,0x18,0x18,0x18,0x00}, { 0x07,0x0C,0x0C,0x38,0x0C,0x0C,0x07,0x00}, { 0x6E,0x3B,0x00,0x00,0x00,0x00,0x00,0x00}, { 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
};

// ---------------------------------------------------------------------------
// Config / state
// ---------------------------------------------------------------------------
// Two sizes, because they answer two different questions. The --shot FRAME is 2560x1440 by default, the Direct3D
// viewer's size, so the two programs' frames are the same shape and the release gate can compare them byte for byte;
// --size changes it. The WINDOW opens at 1280x720: a window has to fit the desktop it opens on, a 2560-wide one is
// bigger than many of them (a Linux desktop does not clamp it as Windows does), and a user who wants it larger drags
// it - the swapchain follows the window. --size, when given, sets both.
static int  FRAME_WIDTH   = 2560;
static int  FRAME_HEIGHT  = 1440;
static int  WINDOW_WIDTH  = 1280;
static int  WINDOW_HEIGHT = 720;
// The overlay strip: the other viewer's OVL_W x OVL_H at FONT_SCALE, so the two strips are the same size in pixels,
// the same four lines and the same advance: 16 pixels a character across a 2560-wide frame. In a narrower window
// the strip is drawn scaled down to the window's width (record_overlay_draw), so the text stays whole and smaller
// rather than being cut off at the right.
static const int OVL_W = 2560, OVL_H = 84, FONT_SCALE = 2, LINE_ADV = 20;
// The help page (key F1): every key this viewer handles, drawn below the strip in the same image and by the same
// rasteriser. It is the Direct3D viewer's page with this viewer's own keys - K, and R's three shaders - and any key
// closes it.
static const char* HELP_LINES[] = {
    "Help - any key closes this page",
    "",
    "Camera (hold):",
    "  Arrows      move the quad or cube left / right / up / down",
    "  W / S       zoom in / out",
    "  A / D       yaw",
    "  Q / E       pitch",
    "  Shift       with any of the above: one third the speed",
    "  Space       reset: the camera, the shown texture, and keys 1 2 V 5-8 off",
    "Display:",
    "  C           quad / cube",
    "  N           next output texture of a material (cycles)",
    "  1           show level 0's texture as stored (toggle)",
    "  2           show level 1's texture as stored (toggle)",
    "  V           renormalise the shown texture as a tangent-space normal (toggle)",
    "Sampling:",
    "  P / B / T   point / bilinear / trilinear filter",
    "  X           anisotropic filtering on/off (applies in trilinear only)",
    "  M           mips on/off (off: the sampler's maxLod 0, every fetch reads mip 0)",
    "  L           level 1's LOD shift on/off (its gradients x 2^lod_bias_level1)",
    "  4           level 0 from the load-time BC4/BC5 pack on/off (uncompressed level 0 only)",
    "Decode:",
    "  K           cooperative-vector decode on/off (only where the device has the extension)",
    "Other:",
    "  5 6 7 8     the shader's spare debug constants (unused by the shipped shaders)",
    "  R           reload the shaders (view.vert, view.frag, view_coopvec.frag)",
    "  F1          this page",
    "  Esc         quit (with this page open: close the page only)",
};
static const int HELP_N = (int)(sizeof(HELP_LINES) / sizeof(HELP_LINES[0]));
static const int OVL_TEX_H = OVL_H + 8 + HELP_N * LINE_ADV;   // the image holds the strip and the help page below it
static bool g_help = false;    // F1: the help page is shown
static int  g_help_eat = 0;    // the key that closed the help page: ignored by the held-key scan until it is released
// Where the stage-4 decode-path token is drawn on the state line: twelve characters in from the right edge of the
// strip, which is past the end of anything the four shared lines can print (the longest is about 110 characters of
// the 160 that fit). tests/run_checks.py compares the two viewers' strips over the columns before it.
static const int COOP_COLUMN = OVL_W - 12 * 8 * FONT_SCALE;
// The depth format is CHOSEN and not assumed, though the choice is a formality: the spec's required-format table
// makes D32_SFLOAT and D16_UNORM both mandatory as optimally-tiled depth-stencil attachments, and X8_D24_UNORM_PACK32
// is the OPTIONAL one of the three. So the first candidate is already guaranteed and the list below is a fallback that
// should never be reached; it is kept because a format query costs one call and turns a driver that fails its own
// guarantee into a refusal by name instead of a failed image creation several steps later.
// pick_depth_format fills this in from the chosen device's own format properties, once, and prints what it took.
static VkFormat g_depth_format = VK_FORMAT_D32_SFLOAT;
static const float CLEAR_COLOUR[4] = { 0.2f, 0.2f, 0.2f, 1.0f };
static const float FOV_DEGREES = 90.0f;      // the other viewer's camera, constant for constant
static const int   FILTER_STATES = 4;        // point, bilinear, trilinear, trilinear with anisotropy
static const float ANISO_MAX = 8.0f;         // the anisotropy asked for when it is on; one line to change

// The window's own quit flag lives in the platform block below, because only a window can raise it; what the frame
// loop reads is win_should_quit.
bool g_resized = false;   // a WM_SIZE that the frame loop turns into one recreate_swapchain
bool g_shot = false;      // --shot: one headless frame, no window, no surface, no keyboard
// Esc, raised by the key handling rather than by the window, because which key quits is the viewer's business and not
// the platform's. Each platform's win_should_quit reads it beside its own close message.
bool g_quit_requested = false;

#include "platform.h"   // the window, the keys and the clock: Win32 in platform_win32.cpp, GLFW in platform_linux.cpp

// ---------------------------------------------------------------------------
// The Vulkan objects. One of everything: this is a viewer with one frame in flight by design (the plan's section
// 1.12), so there is no per-frame duplication of anything and every object below is created once.
// ---------------------------------------------------------------------------
static VkInstance       g_instance = VK_NULL_HANDLE;
static VkPhysicalDevice g_phys = VK_NULL_HANDLE;
static VkDevice         g_device = VK_NULL_HANDLE;
static VkQueue          g_queue = VK_NULL_HANDLE;
static uint32_t         g_queue_family = 0;
static VkSurfaceKHR     g_surface = VK_NULL_HANDLE;
static VkSwapchainKHR   g_swapchain = VK_NULL_HANDLE;
static VkFormat         g_swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;
static VkExtent2D       g_swapchain_extent = { 0, 0 };
static std::vector<VkImage>     g_swapchain_images;
static std::vector<VkImageView> g_swapchain_views;
// One render-finished semaphore PER SWAPCHAIN IMAGE rather than per frame. A semaphore signalled for presentation
// cannot be waited on again until that image comes back from the presentation engine, and reusing one per frame is the
// classic validation error in a program shaped exactly like this one.
static std::vector<VkSemaphore> g_render_finished;
static VkImage          g_depth_image = VK_NULL_HANDLE;
static VkDeviceMemory   g_depth_memory = VK_NULL_HANDLE;
static VkImageView      g_depth_view = VK_NULL_HANDLE;
static VkCommandPool    g_command_pool = VK_NULL_HANDLE;
static VkCommandBuffer  g_command_buffer = VK_NULL_HANDLE;
static VkSemaphore      g_image_available = VK_NULL_HANDLE;
static VkFence          g_frame_fence = VK_NULL_HANDLE;

// What the chosen device can do, read once at device creation and acted on rather than assumed.
static bool  g_has_bc = false;          // textureCompressionBC: without it a BC4 / BC5 asset is refused by name
static bool  g_has_aniso = false;       // samplerAnisotropy: without it key X's state is permanently off
static float g_timestamp_period = 0.0f; // nanoseconds a timestamp tick is worth, which --bench turns ticks into
static uint32_t g_timestamp_bits = 0;   // the queue family's valid timestamp bits; 0 means it cannot be measured here

// ---------------------------------------------------------------------------
// Cooperative vectors: STAGE 4, and the whole of what this viewer knows about VK_NV_cooperative_vector.
//
// The decode is one matrix-vector product of at most 18 by 24 per pixel, which is exactly the shape that extension
// addresses, so the experiment is worth making. But the extension is ONE VENDOR'S, has no portable successor and the
// Direct3D preview of the same idea was withdrawn, so the path is permanently OPTIONAL: the plain GLSL decode is the
// product and is what runs everywhere, and this is a second pipeline chosen only when a four-part device query passes.
// On a device that fails the query the viewer says so in one line and never mentions the extension again.
//
// The padding is what makes the shape a compile-time constant. The shading language requires M, K and the three
// interpretation arguments to be constant expressions, and this format's nin and nout vary per asset, so the matrix is
// ALWAYS the padded 18 by 24, with zero rows past nout and zero columns past nin, and phi is zero past nin. A zero row
// contributes an output that is already ignored and a zero column multiplies a zero feature, so the padded product is
// the unpadded one in exact arithmetic - and every measurement is then of one shape and comparable across assets.
// ---------------------------------------------------------------------------
#if NNTC_HAVE_COOPVEC
static const uint32_t CV_M = 18, CV_K = 24;   // the padded shape, and the literals the second shader is written for
#endif

static bool  g_cv_available = false;    // the device query passed, so the second pipeline exists
static std::string g_cv_why_not;        // and when it did not, which part of it failed
static bool  g_cv_on = false;           // which of the two pipelines the next frame binds (key K, --coopvec)
static int   g_cv_want = -1;            // --coopvec: -1 not given, 0 off, 1 on (refused when it is unavailable)
static bool  g_cv_fp32 = false;         // the chosen tuple accumulates in fp32; false is the all-fp16 fallback
#if NNTC_HAVE_COOPVEC
static bool  g_cv_need_float16 = false; // and that fallback needs the shaderFloat16 feature on the device
static PFN_vkGetPhysicalDeviceCooperativeVectorPropertiesNV g_cv_get_properties = nullptr;
static PFN_vkConvertCooperativeVectorMatrixNV g_cv_convert = nullptr;
#endif
static VkBuffer       g_cv_buffer = VK_NULL_HANDLE;   // the converted matrix and the bias, in one storage buffer
static VkDeviceMemory g_cv_memory = VK_NULL_HANDLE;
static void*          g_cv_mapped = nullptr;
static VkDeviceSize   g_cv_bias_offset = 0;           // where the bias starts in it, in bytes; the matrix is at 0
#if NNTC_HAVE_COOPVEC
static VkDeviceSize   g_cv_bytes = 0;         // how large that buffer is, which only the code that fills it reads
#endif

#if NNTC_HAVE_COOPVEC
// One fp32 to fp16 conversion, round to nearest even, for the bias under the all-fp16 tuple. It is written out
// rather than taken from a library because it is eighteen numbers once at load, and because the rounding rule has
// to be the same one vkConvertCooperativeVectorMatrixNV applies to the matrix beside it. Subnormals and the
// overflow to infinity are both handled: a decoder bias past 65504 would otherwise become a quiet zero.
static uint16_t float_to_half(float f) {
    uint32_t bits = 0;
    memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exponent = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mantissa = bits & 0x7FFFFFu;
    if (((bits >> 23) & 0xFFu) == 0xFFu)                     // an infinity or a NaN stays one
        return (uint16_t)(sign | 0x7C00u | (mantissa ? 0x200u : 0u));
    if (exponent >= 0x1F) return (uint16_t)(sign | 0x7C00u);  // too large for fp16: infinity
    if (exponent <= 0) {
        if (exponent < -10) return (uint16_t)sign;             // too small even to be subnormal
        const uint32_t m = mantissa | 0x800000u;               // the implicit one, then shifted into subnormal
        const int shift = 14 - exponent;
        const uint32_t half = m >> shift;
        const uint32_t rest = m & ((1u << shift) - 1u), halfway = 1u << (shift - 1);
        return (uint16_t)(sign | (half + ((rest > halfway || (rest == halfway && (half & 1u))) ? 1u : 0u)));
    }
    const uint32_t half = ((uint32_t)exponent << 10) | (mantissa >> 13);
    const uint32_t rest = mantissa & 0x1FFFu;
    return (uint16_t)(sign | (half + ((rest > 0x1000u || (rest == 0x1000u && (half & 1u))) ? 1u : 0u)));
}
#endif

// --bench N: N headless frames with a timestamp either side of the scene draw. It is the only honest way to time this
// program, because the frame loop waits on the device by design, so a wall clock would measure the wait.
static int g_bench_frames = 0;
static VkQueryPool g_query_pool = VK_NULL_HANDLE;   // two timestamps, and null in every mode but --bench

// NDEBUG rather than _DEBUG: _DEBUG is MSVC's own macro and is not defined by gcc or clang, so a Debug build on Linux
// would silently have had no validation layer at all. NDEBUG is the standard's, set by every Release configuration and
// absent from every Debug one, so the layer is on wherever a Debug build is.
#ifndef NDEBUG
#define ENABLE_VALIDATION 1
#else
#define ENABLE_VALIDATION 0
#endif

#if ENABLE_VALIDATION
static VkDebugUtilsMessengerEXT g_messenger = VK_NULL_HANDLE;
#endif

// The memory type a requirement and a set of property flags pick out. This is the whole of what an allocator library
// would do for this program: a viewer makes perhaps ten allocations in its life, and the guaranteed floor on
// vkAllocateMemory calls is 4096.
// The same search without the refusal, for a caller that has a second choice: UINT32_MAX means "no such type here".
static uint32_t find_memory_type_or_none(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(g_phys, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

static uint32_t find_memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    const uint32_t i = find_memory_type_or_none(bits, want);
    if (i != UINT32_MAX) return i;
    fprintf(stderr, "ERROR: no memory type with the properties this allocation needs (0x%x of mask 0x%x)\n",
            (unsigned)want, (unsigned)bits);
    exit(1);
}

#if ENABLE_VALIDATION
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                     VkDebugUtilsMessageTypeFlagsEXT,
                                                     const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    // A validation message is not this program's error path - it is the layer's - so it is reported and the frame
    // carries on. It goes to stderr because it is a diagnostic and because a gate that greps a frame's stdout must not
    // have to filter it out.
    if (severity & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT))
        fprintf(stderr, "validation: %s\n", data && data->pMessage ? data->pMessage : "(no message)");
    return VK_FALSE;
}

static bool instance_layer_present(const char* name) {
    uint32_t n = 0;
    if (vkEnumerateInstanceLayerProperties(&n, nullptr) != VK_SUCCESS || !n) return false;
    std::vector<VkLayerProperties> layers(n);
    if (vkEnumerateInstanceLayerProperties(&n, layers.data()) != VK_SUCCESS) return false;
    for (const VkLayerProperties& l : layers) if (!strcmp(l.layerName, name)) return true;
    return false;
}

static bool instance_extension_present(const char* name) {
    uint32_t n = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr) != VK_SUCCESS || !n) return false;
    std::vector<VkExtensionProperties> exts(n);
    if (vkEnumerateInstanceExtensionProperties(nullptr, &n, exts.data()) != VK_SUCCESS) return false;
    for (const VkExtensionProperties& e : exts) if (!strcmp(e.extensionName, name)) return true;
    return false;
}
#endif   // ENABLE_VALIDATION: the three above exist for the layer and nothing else, and a release build has no layer

// The instance. `windowed` decides the extension list: the headless shot asks for no surface extension at all, which
// is what lets it run on a machine with no desktop and no compositor.
static void create_instance(bool windowed) {
    uint32_t loader_version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion(&loader_version) != VK_SUCCESS) loader_version = VK_API_VERSION_1_0;
    if (loader_version < VK_API_VERSION_1_3) {
        // "no Vulkan device" is the phrase the release gate reads to tell a machine that cannot run this viewer from a
        // viewer that is broken, and a skipped arm must never be recorded as a passed one. So all three refusals that
        // mean "not here" carry it: this one, an enumeration that returns nothing, and one whose devices are all
        // unusable.
        fprintf(stderr, "ERROR: no Vulkan device: this machine's Vulkan loader offers %u.%u and this viewer needs "
                        "1.3; update the loader (the graphics driver installs it) and try again\n",
                VK_API_VERSION_MAJOR(loader_version), VK_API_VERSION_MINOR(loader_version));
        exit(1);
    }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "nntc_view_vk"; app.applicationVersion = 1;
    app.pEngineName = "nntc_view_vk"; app.engineVersion = 1;
    app.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> extensions;
    if (windowed && !win_instance_extensions(extensions)) {
        fprintf(stderr, "ERROR: this build has no window system, so no surface extension to ask for\n");
        exit(1);
    }
    std::vector<const char*> layers;
#if ENABLE_VALIDATION
    // Like the Direct3D viewer's D3D11_CREATE_DEVICE_DEBUG, a missing validation layer is a warning and not a failure:
    // a Debug build must still run on a machine with no SDK installed.
    const bool validation = instance_layer_present("VK_LAYER_KHRONOS_validation");
    const bool debug_utils = validation && instance_extension_present(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (validation) layers.push_back("VK_LAYER_KHRONOS_validation");
    else fprintf(stderr, "WARNING: VK_LAYER_KHRONOS_validation is not installed; running without it\n");
    if (debug_utils) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = (uint32_t)extensions.size();
    ci.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
    ci.enabledLayerCount = (uint32_t)layers.size();
    ci.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
    VK_CHECK(vkCreateInstance(&ci, nullptr, &g_instance));

#if ENABLE_VALIDATION
    if (debug_utils) {
        // The messenger's two entry points are extension functions, so they are fetched rather than linked. Two
        // vkGetInstanceProcAddr calls is the whole reason a meta-loader exists, and at two it is not a reason.
        auto create = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(g_instance,
                                                                                "vkCreateDebugUtilsMessengerEXT");
        if (create) {
            VkDebugUtilsMessengerCreateInfoEXT mi{};
            mi.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            mi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            mi.pfnUserCallback = debug_callback;
            VK_CHECK(create(g_instance, &mi, nullptr, &g_messenger));
        }
        printf("the validation layer is active\n");
    }
#endif
}

// The queue family a device can draw with, and present to `surface` when there is one. VK_QUEUE_FAMILY_IGNORED means
// the device has none, which is how a compute-only or display-less device is passed over.
static uint32_t graphics_family(VkPhysicalDevice phys, VkSurfaceKHR surface) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, nullptr);
    std::vector<VkQueueFamilyProperties> families(n);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, families.data());
    for (uint32_t i = 0; i < n; i++) {
        if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
        if (surface != VK_NULL_HANDLE) {
            VkBool32 present = VK_FALSE;
            if (vkGetPhysicalDeviceSurfaceSupportKHR(phys, i, surface, &present) != VK_SUCCESS || !present) continue;
        }
        return i;
    }
    return VK_QUEUE_FAMILY_IGNORED;
}

// Declared here and defined below beside the cooperative-vector query, which is the other caller: the device list
// needs it to say "no VK_KHR_swapchain" about a device that could otherwise draw.
static bool device_extension_present(VkPhysicalDevice phys, const char* name, uint32_t* revision = nullptr);

// Everything a physical device has to be for this viewer to draw on it, in one place so that the device list and
// --device N refuse for the same reasons in the same words. `surface` is VK_NULL_HANDLE in the headless mode, which
// presents nothing, so VK_KHR_swapchain is asked of a device only when there is a window to present to.
// Returns nullptr when the device is usable, and the reason it is not otherwise.
//
// The first refusal is TWO refusals wearing one word, and they are told apart here because they mean opposite things
// to whoever reads the line: a device with no graphics queue at all cannot draw anything (a compute-only part), while
// a device that draws but has no family able to present to THIS surface is a perfectly good GPU that the window is
// simply not on - a headless --shot on it still works.
static const char* device_unsuitable(VkPhysicalDevice phys, const VkPhysicalDeviceProperties& p, VkSurfaceKHR surface) {
    if (graphics_family(phys, surface) == VK_QUEUE_FAMILY_IGNORED)
        return surface != VK_NULL_HANDLE && graphics_family(phys, VK_NULL_HANDLE) != VK_QUEUE_FAMILY_IGNORED
             ? "a graphics queue, but no family that can present to this surface" : "no graphics queue";
    if (p.apiVersion < VK_API_VERSION_1_3) return "below Vulkan 1.3";
    // A queue that reports presentation support and a device that does not expose the swapchain extension is not a
    // contradiction: presentation support is a queue-family property and the extension is what create_device would
    // have to enable to use it. Asking now means the refusal names the missing extension instead of arriving as a
    // vkCreateDevice failure several steps later.
    if (surface != VK_NULL_HANDLE && !device_extension_present(phys, VK_KHR_SWAPCHAIN_EXTENSION_NAME, nullptr))
        return "no VK_KHR_swapchain";
    return nullptr;
}

// The depth attachment's format, chosen once from what the CHOSEN device can actually do with optimal tiling, in
// order of preference: the 32-bit float this was developed on, then the packed 24-bit one, then the 16-bit one. A
// depth format is not a free choice - an image created in a format whose optimalTilingFeatures lack
// DEPTH_STENCIL_ATTACHMENT_BIT is out of spec - so the property is asked for rather than assumed. D32_SFLOAT and
// D16_UNORM are both in the spec's required-format table for this use (X8_D24_UNORM_PACK32 is the optional one), so
// the first candidate already has to be there and the other two exist for a driver that does not keep that promise.
static void pick_depth_format(void) {
    static const struct { VkFormat format; const char* name; } candidates[] = {
        { VK_FORMAT_D32_SFLOAT, "D32_SFLOAT" },
        { VK_FORMAT_X8_D24_UNORM_PACK32, "X8_D24_UNORM_PACK32" },
        { VK_FORMAT_D16_UNORM, "D16_UNORM" },
    };
    for (const auto& c : candidates) {
        VkFormatProperties fp{};
        vkGetPhysicalDeviceFormatProperties(g_phys, c.format, &fp);
        if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            g_depth_format = c.format;
            printf("depth format: %s\n", c.name);
            return;
        }
    }
    fprintf(stderr, "ERROR: this device offers none of D32_SFLOAT, X8_D24_UNORM_PACK32 or D16_UNORM as an "
                    "optimally-tiled depth attachment\n");
    exit(1);
}

static const char* device_kind(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
        default: return "other";
    }
}

// The standard choice: the first discrete GPU with a queue family that can draw and present to this surface, else the
// first device that has one at all. `want` is --device N and overrides the choice outright, because the whole point of
// the flag is to exercise a device the rule would pass over - the integrated part beside a discrete one, say. The name
// of what was chosen is printed either way, so there is never any doubt which GPU drew the frame.
static void pick_physical_device(int want) {
    uint32_t n = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(g_instance, &n, nullptr));
    if (!n) { fprintf(stderr, "ERROR: no Vulkan device was found\n"); exit(1); }
    std::vector<VkPhysicalDevice> devices(n);
    VK_CHECK(vkEnumeratePhysicalDevices(g_instance, &n, devices.data()));

    int chosen = -1;
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devices[i], &p);
        const char* why_not = device_unsuitable(devices[i], p, g_surface);
        printf("device %u: %s (%s, Vulkan %u.%u)", i, p.deviceName, device_kind(p.deviceType),
               VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion));
        // The disqualifier is printed after two spaces and a dash, which is the shape tests/run_checks.py reads to
        // tell a device it may ask for with --device from one it may not.
        printf("%s%s\n", why_not ? "  - " : "", why_not ? why_not : "");
        if (why_not) continue;
        if (chosen < 0) chosen = (int)i;
        else {
            VkPhysicalDeviceProperties c{};
            vkGetPhysicalDeviceProperties(devices[(size_t)chosen], &c);
            if (c.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
                p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) chosen = (int)i;
        }
    }
    if (want >= 0) {
        if (want >= (int)n) {
            fprintf(stderr, "ERROR: --device %d, and this machine has %u Vulkan device(s)\n", want, n);
            exit(1);
        }
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devices[(size_t)want], &p);
        if (graphics_family(devices[(size_t)want], g_surface) == VK_QUEUE_FAMILY_IGNORED) {
            fprintf(stderr, "ERROR: --device %d (%s) has no queue family that can %s\n", want, p.deviceName,
                    g_surface != VK_NULL_HANDLE ? "draw and present to this window" : "draw");
            exit(1);
        }
        if (p.apiVersion < VK_API_VERSION_1_3) {
            fprintf(stderr, "ERROR: --device %d (%s) offers Vulkan %u.%u and this viewer needs 1.3\n", want,
                    p.deviceName, VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion));
            exit(1);
        }
        if (g_surface != VK_NULL_HANDLE &&
            !device_extension_present(devices[(size_t)want], VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
            fprintf(stderr, "ERROR: --device %d (%s) does not expose %s, so it cannot present to a window\n", want,
                    p.deviceName, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
            exit(1);
        }
        chosen = want;
    }
    if (chosen < 0) {
        // Two different machines end up here and they are not the same news. "no Vulkan device" is the phrase the
        // release gate reads to SKIP its Vulkan arms rather than fail them, so it is kept for the case it was minted
        // for: this machine cannot run the viewer at all, because nothing here is 1.3. A machine whose devices are
        // 1.3 and draw, and only cannot present to this particular surface, is a different story - the headless
        // --shot on it works - so it gets its own words and the gate reads it as a real refusal.
        for (uint32_t i = 0; i < n; i++) {
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(devices[i], &p);
            if (p.apiVersion < VK_API_VERSION_1_3) continue;
            if (graphics_family(devices[i], VK_NULL_HANDLE) == VK_QUEUE_FAMILY_IGNORED) continue;
            fprintf(stderr, "ERROR: this machine's Vulkan device(s) can draw and none of them has a queue family that "
                            "can present to this window; --shot renders headless on the same device\n");
            exit(1);
        }
        fprintf(stderr, "ERROR: no Vulkan device: none of them is 1.3 with a queue family that can %s\n",
                g_surface != VK_NULL_HANDLE ? "draw and present to this window" : "draw");
        exit(1);
    }
    g_phys = devices[(size_t)chosen];
    g_queue_family = graphics_family(g_phys, g_surface);
    VkPhysicalDeviceProperties p{};
    vkGetPhysicalDeviceProperties(g_phys, &p);
    printf("drawing on device %d: %s (%s, Vulkan %u.%u), queue family %u\n", chosen, p.deviceName,
           device_kind(p.deviceType), VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion),
           g_queue_family);
    // What --bench needs, read here because both halves of it belong to the chosen device and its queue family: the
    // nanoseconds a tick is worth, and how many bits of the counter are meaningful. A family may report zero, which
    // means the queue cannot be timed at all and --bench says so rather than printing a number made of nothing.
    g_timestamp_period = p.limits.timestampPeriod;
    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &families, nullptr);
    std::vector<VkQueueFamilyProperties> family_props(families);
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &families, family_props.data());
    if (g_queue_family < families) g_timestamp_bits = family_props[g_queue_family].timestampValidBits;
    // The depth format belongs to the device that was just chosen and to nothing before it, so it is settled here
    // rather than at the top of the file.
    pick_depth_format();
}

// The frame's size against the chosen device's own limits, asked once the device is known and before an image of that
// size is created. A colour attachment has to fit maxImageDimension2D and the render area has to fit
// maxFramebufferWidth / Height; --size accepts anything up to 16384, which is above what some parts offer, and an
// image created past the limit is out of spec rather than clamped.
static void check_target_size(void) {
    VkPhysicalDeviceProperties p{};
    vkGetPhysicalDeviceProperties(g_phys, &p);
    const uint32_t w = (uint32_t)FRAME_WIDTH, h = (uint32_t)FRAME_HEIGHT;
    if (w > p.limits.maxImageDimension2D || h > p.limits.maxImageDimension2D) {
        fprintf(stderr, "ERROR: a %ux%u frame is asked for and this device's maxImageDimension2D is %u\n",
                w, h, p.limits.maxImageDimension2D);
        exit(1);
    }
    if (w > p.limits.maxFramebufferWidth || h > p.limits.maxFramebufferHeight) {
        fprintf(stderr, "ERROR: a %ux%u frame is asked for and this device's maxFramebufferWidth / Height is %u / %u\n",
                w, h, p.limits.maxFramebufferWidth, p.limits.maxFramebufferHeight);
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// The cooperative-vector device query: four parts, and every one of them has to pass before a single line of the
// second pipeline is built. The order below is the order the extension's own documentation gives, and none of the four
// is a formality:
//
//   1. the extension is in the device's list at all. It is NVIDIA-only today, so most devices stop here;
//   2. the cooperativeVector FEATURE is on. An extension can be advertised with its feature off (the training feature
//      is a second boolean and this viewer never needs it);
//   3. cooperativeVectorSupportedStages contains the FRAGMENT stage. The shading language permits every stage
//      "subject to api-specific limitations" and there is no compile-time check for it, so a driver that offered the
//      extension for compute only would compile the shader happily and fail at pipeline creation;
//   4. a supported TYPE TUPLE. vkGetPhysicalDeviceCooperativeVectorPropertiesNV enumerates combinations of input type,
//      the three interpretations, result type and transpose - there are no size fields in that struct, so it says
//      nothing about M and K, and the sizes are covered by maxCooperativeVectorComponents instead.
//
// A fifth is this program's own rather than the extension's, and it is here because it is found at the same moment:
// glslang compiles a cooperative-vector module with OpCapability VulkanMemoryModel, so the device has to offer the
// vulkanMemoryModel feature or the module cannot be used. It is core 1.2 and every driver that has this extension has
// it, but a capability a shader declares and a device does not enable is undefined behaviour, not a slow path.
// ---------------------------------------------------------------------------
#if NNTC_HAVE_COOPVEC
static const char* component_type_name(VkComponentTypeKHR t) {
    switch (t) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return "fp16";
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return "fp32";
        case VK_COMPONENT_TYPE_FLOAT64_KHR: return "fp64";
        case VK_COMPONENT_TYPE_SINT8_KHR: return "sint8";
        case VK_COMPONENT_TYPE_SINT16_KHR: return "sint16";
        case VK_COMPONENT_TYPE_SINT32_KHR: return "sint32";
        case VK_COMPONENT_TYPE_UINT8_KHR: return "uint8";
        case VK_COMPONENT_TYPE_UINT16_KHR: return "uint16";
        case VK_COMPONENT_TYPE_UINT32_KHR: return "uint32";
        case VK_COMPONENT_TYPE_SINT8_PACKED_NV: return "sint8packed";
        case VK_COMPONENT_TYPE_UINT8_PACKED_NV: return "uint8packed";
        case VK_COMPONENT_TYPE_FLOAT_E4M3_NV: return "e4m3";
        case VK_COMPONENT_TYPE_FLOAT_E5M2_NV: return "e5m2";
        default: return "?";
    }
}
#endif

// The revision is reported rather than the header's own constant: what matters is what the DRIVER implements, and the
// two are not the same number on a machine whose SDK is newer than its driver.
static bool device_extension_present(VkPhysicalDevice phys, const char* name, uint32_t* revision) {
    uint32_t n = 0;
    if (vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr) != VK_SUCCESS || !n) return false;
    std::vector<VkExtensionProperties> exts(n);
    if (vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, exts.data()) != VK_SUCCESS) return false;
    for (const VkExtensionProperties& e : exts)
        if (!strcmp(e.extensionName, name)) { if (revision) *revision = e.specVersion; return true; }
    return false;
}

// Sets g_cv_available and, when it is false, the reason. It prints exactly one line either way, so a reader of any run
// can say which decode path the frame could have taken without reading this file.
// `extension_present`, `revision`, `f12` and `cvf` are what create_device's single vkGetPhysicalDeviceFeatures2 query
// already learned, passed in rather than asked for again: one query for every feature this program cares about keeps
// the cooperative-vector structure out of the chain on a device that does not have the extension, which is where
// chaining it would be a question the layer is entitled to object to.
#if NNTC_HAVE_COOPVEC
static void query_cooperative_vector(bool extension_present, uint32_t revision,
                                     const VkPhysicalDeviceVulkan12Features& f12,
                                     const VkPhysicalDeviceCooperativeVectorFeaturesNV& cvf) {
    g_cv_available = false;
    g_cv_why_not = "the device does not offer " VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME;
    if (!extension_present) {
        printf("cooperative vectors: not available - %s\n", g_cv_why_not.c_str());
        return;
    }

    if (!cvf.cooperativeVector) {
        g_cv_why_not = "the extension is present and its cooperativeVector feature is off";
        printf("cooperative vectors: not available - %s\n", g_cv_why_not.c_str());
        return;
    }
    if (!f12.vulkanMemoryModel) {
        g_cv_why_not = "this device has no vulkanMemoryModel feature, which a cooperative-vector module declares";
        printf("cooperative vectors: not available - %s\n", g_cv_why_not.c_str());
        return;
    }

    VkPhysicalDeviceCooperativeVectorPropertiesNV cvp{};
    cvp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_PROPERTIES_NV;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &cvp;
    vkGetPhysicalDeviceProperties2(g_phys, &p2);
    if (!(cvp.cooperativeVectorSupportedStages & VK_SHADER_STAGE_FRAGMENT_BIT)) {
        g_cv_why_not = "the fragment stage is not in cooperativeVectorSupportedStages, and the decode is a fragment "
                       "shader";
        printf("cooperative vectors: not available - %s\n", g_cv_why_not.c_str());
        return;
    }
    if (cvp.maxCooperativeVectorComponents < CV_K) {
        char why[160];
        snprintf(why, sizeof(why), "maxCooperativeVectorComponents is %u and the padded shape needs %u",
                 cvp.maxCooperativeVectorComponents, (unsigned)CV_K);
        g_cv_why_not = why;
        printf("cooperative vectors: not available - %s\n", g_cv_why_not.c_str());
        return;
    }

    g_cv_get_properties = (PFN_vkGetPhysicalDeviceCooperativeVectorPropertiesNV)
        vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceCooperativeVectorPropertiesNV");
    if (!g_cv_get_properties) {
        g_cv_why_not = "vkGetPhysicalDeviceCooperativeVectorPropertiesNV was not found in the loader";
        printf("cooperative vectors: not available - %s\n", g_cv_why_not.c_str());
        return;
    }
    uint32_t n = 0;
    if (g_cv_get_properties(g_phys, &n, nullptr) != VK_SUCCESS || !n) {
        g_cv_why_not = "the device enumerates no cooperative-vector type tuple at all";
        printf("cooperative vectors: not available - %s\n", g_cv_why_not.c_str());
        return;
    }
    std::vector<VkCooperativeVectorPropertiesNV> tuples(n);
    for (VkCooperativeVectorPropertiesNV& t : tuples) t.sType = VK_STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV;
    if (g_cv_get_properties(g_phys, &n, tuples.data()) != VK_SUCCESS) {
        g_cv_why_not = "vkGetPhysicalDeviceCooperativeVectorPropertiesNV failed on its second call";
        printf("cooperative vectors: not available - %s\n", g_cv_why_not.c_str());
        return;
    }
    // The tuples this viewer can use, in the order it would rather have them. Two things are the same in both and are
    // the whole reason this path exists at all: the MATRIX is read as fp16 - W is fp32 in the asset and quantising it
    // further would be a format change made on the viewer's side of the fence - and nothing is transposed, because the
    // format publishes W row-major and the extension's row-major semantics are the same order.
    //
    // What differs is where the arithmetic lands. The first tuple keeps the bias and the RESULT in fp32, so the
    // accumulation is exact; its input vector is fp32 but is read through an fp16 input INTERPRETATION (the second
    // field, which the shader's third argument matches), so phi is rounded to fp16 on the way in and the difference
    // from the plain path is the 11-bit mantissa of the weights and of the inputs both. The second is all-fp16, which is
    // what NVIDIA's driver actually enumerates on this machine, and it accumulates at the implementation's own
    // precision rather than at fp32. Which one a run took is printed, because the precision numbers mean different
    // things under the two.
    //
    // The transpose flag is DELIBERATELY not matched on. It says whether an opaque-layout matrix of these types also
    // supports transposition, which is a capability and not a requirement, and this decode never transposes anything.
    struct Tuple { VkComponentTypeKHR input, input_interp, matrix_interp, bias_interp, result; bool fp32; };
    const Tuple wanted[2] = {
        { VK_COMPONENT_TYPE_FLOAT32_KHR, VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT16_KHR,
          VK_COMPONENT_TYPE_FLOAT32_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR, true },
        { VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT16_KHR,
          VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT16_KHR, false },
    };
    bool found = false;
    for (const Tuple& want_tuple : wanted) {
        // The all-fp16 form declares the Float16 capability in the module, so a device that cannot offer
        // shaderFloat16 cannot run it and the search moves on rather than building a pipeline that is undefined.
        if (!want_tuple.fp32 && !f12.shaderFloat16) continue;
        for (const VkCooperativeVectorPropertiesNV& t : tuples)
            if (t.inputType == want_tuple.input && t.inputInterpretation == want_tuple.input_interp &&
                t.matrixInterpretation == want_tuple.matrix_interp && t.biasInterpretation == want_tuple.bias_interp &&
                t.resultType == want_tuple.result) { found = true; break; }
        if (found) { g_cv_fp32 = want_tuple.fp32; g_cv_need_float16 = !want_tuple.fp32; break; }
    }
    if (!found) {
        // The table is printed only when the tuple is missing, which is the one moment a reader needs it: this is the
        // fallback of the plan's risk 10, and "the path is built only for a tuple that is actually present" is only
        // worth anything if the absent one can be seen.
        printf("cooperative vectors: not available - this device enumerates %u type tuple(s) and none of them is\n"
               "  an fp16 matrix with an fp32 or an fp16 bias and result. What it does offer:\n", n);
        for (const VkCooperativeVectorPropertiesNV& t : tuples)
            printf("    input %s as %s x matrix %s + bias %s -> %s%s\n", component_type_name(t.inputType),
                   component_type_name(t.inputInterpretation), component_type_name(t.matrixInterpretation),
                   component_type_name(t.biasInterpretation), component_type_name(t.resultType),
                   t.transpose ? ", transposed" : "");
        g_cv_why_not = "no fp16-matrix type tuple this viewer can use is enumerated";
        return;
    }
    g_cv_available = true;
    g_cv_why_not.clear();
    printf("cooperative vectors: available - %s revision %u, the fragment stage, %u components, %s "
           "(%u tuples enumerated)\n", VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME, revision,
           cvp.maxCooperativeVectorComponents,
           g_cv_fp32 ? "input fp32 as fp16 x matrix fp16 + bias fp32 -> fp32"
                     : "input fp16 x matrix fp16 + bias fp16 -> fp16 (this device enumerates no fp32 result, so the "
                       "accumulation is the implementation's own and not fp32)",
           n);
}
#else
// The same function where the headers have no such extension to ask about. It takes no arguments because there is
// nothing to pass: the refusal is a property of the build and not of the device, and it is printed in the same
// shape and the same one line as every other refusal above, so a run's own output still says which decode path it
// could have taken. g_cv_available stays false, which is what every branch downstream already reads.
static void query_cooperative_vector(void) {
    g_cv_available = false;
    g_cv_why_not = "this build's Vulkan headers (version " NNTC_STRINGIFY(VK_HEADER_VERSION) ", from "
                   "VK_HEADER_VERSION) predate VK_NV_cooperative_vector, which arrived in 1.4.307; the plain "
                   "path is used. It is an NVIDIA extension: AMD and Intel devices do not offer it, so on those "
                   "nothing is missing";
    printf("cooperative vectors: not available - %s\n", g_cv_why_not.c_str());
}
#endif

// The device, with the two 1.3 booleans this program is built on and the two 1.0 ones the asset needs.
// VK_KHR_swapchain is asked for only in the windowed mode: the headless shot never presents anything.
//
// textureCompressionBC is what makes BC4 and BC5 guaranteed to support linear filtering, which is how the encoder
// fitted the asset; a device without it is not refused here, because an uncompressed asset still renders on it, but
// the per-format check at load then refuses the file by name rather than drawing something wrong. samplerAnisotropy
// is the same shape of thing for key X: without it anisotropy is permanently off and the viewer says so.
static void create_device(bool windowed) {
    VkPhysicalDeviceFeatures have{};
    vkGetPhysicalDeviceFeatures(g_phys, &have);
    g_has_bc = have.textureCompressionBC == VK_TRUE;
    g_has_aniso = have.samplerAnisotropy == VK_TRUE;
    if (!g_has_bc) printf("note: this device does not report textureCompressionBC; a BC4 / BC5 asset will be refused\n");
    if (!g_has_aniso) printf("note: this device does not report samplerAnisotropy; anisotropic filtering is off\n");

    // ONE feature query, for everything this program is built on. VkPhysicalDeviceVulkan13Features is chained into it
    // and its two booleans are required BY NAME even though a device that reports Vulkan 1.3 - which pick_physical_device
    // already insisted on - implies them: the spec says a 1.3 implementation supports dynamic rendering and
    // synchronisation2, and querying rather than assuming is the defensive form the Khronos guidance recommends, so a
    // driver that reports 1.3 and clears one of them is refused by name here instead of drawing nothing later.
    // The cooperative-vector structure joins the chain only where the extension is present, because the features of an
    // extension a device does not have are not a thing to ask it about.
#if NNTC_HAVE_COOPVEC
    uint32_t cv_revision = 0;
    const bool cv_extension = device_extension_present(g_phys, VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME, &cv_revision);
    VkPhysicalDeviceCooperativeVectorFeaturesNV cvf_have{};
    cvf_have.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV;
#endif
    VkPhysicalDeviceVulkan12Features f12_have{};
    f12_have.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
#if NNTC_HAVE_COOPVEC
    f12_have.pNext = cv_extension ? &cvf_have : nullptr;
#endif
    VkPhysicalDeviceVulkan13Features f13_have{};
    f13_have.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f13_have.pNext = &f12_have;
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &f13_have;
    vkGetPhysicalDeviceFeatures2(g_phys, &f2);
    if (!f13_have.dynamicRendering || !f13_have.synchronization2) {
        fprintf(stderr, "ERROR: this device reports Vulkan 1.3 and not %s%s%s, which this viewer draws every frame "
                        "with\n", f13_have.dynamicRendering ? "" : "dynamicRendering",
                (!f13_have.dynamicRendering && !f13_have.synchronization2) ? " or " : "",
                f13_have.synchronization2 ? "" : "synchronization2");
        exit(1);
    }
    printf("features: dynamicRendering and synchronization2 both reported\n");

    // Stage 4, and it happens before the device exists because its answer decides what the device is created with.
#if NNTC_HAVE_COOPVEC
    query_cooperative_vector(cv_extension, cv_revision, f12_have, cvf_have);
#else
    query_cooperative_vector();
#endif
    // --coopvec 1 on a device that has no such thing is a REFUSAL and not a quiet fall back to the plain path: the
    // flag exists to make the second pipeline run, and a run that silently did not is a measurement of the wrong
    // thing. --coopvec 0 is always honoured, because the plain path is the product and runs everywhere.
    if (g_cv_want == 1 && !g_cv_available) {
        fprintf(stderr, "ERROR: --coopvec 1 was asked for and this device cannot: %s\n", g_cv_why_not.c_str());
        exit(1);
    }
    // ON by default when the device has it, because exercising it is the point of the second pipeline; the plain path
    // is one keystroke or one flag away, and every comparison in the gate names the flag rather than relying on this.
    g_cv_on = g_cv_available && g_cv_want != 0;
    // Which path this run will draw with, said once and in the same words the key K prints. A frame whose decode
    // path cannot be read off its own run is a measurement of an unknown thing.
    printf("decode: %s\n", g_cv_on ? "cooperative vectors" : "plain");

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qi{};
    qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qi.queueFamilyIndex = g_queue_family; qi.queueCount = 1; qi.pQueuePriorities = &priority;

    VkPhysicalDeviceFeatures want{};
    want.textureCompressionBC = g_has_bc ? VK_TRUE : VK_FALSE;
    want.samplerAnisotropy = g_has_aniso ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceVulkan13Features f13{};
    f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;

    std::vector<const char*> extensions;
    if (windowed) extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    // The extension and its two features are asked for ONLY when the query passed. That is the plan's amendment 1 in
    // code: on every other device this program creates exactly the device stage 3 created, and a driver that would
    // have refused the extension never sees it named.
#if NNTC_HAVE_COOPVEC
    VkPhysicalDeviceCooperativeVectorFeaturesNV cvf{};
    cvf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV;
    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    if (g_cv_available) {
        extensions.push_back(VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME);
        cvf.cooperativeVector = VK_TRUE;   // and never cooperativeVectorTraining: this viewer only infers
        f12.vulkanMemoryModel = VK_TRUE;   // the capability glslang puts in a cooperative-vector module
        if (g_cv_need_float16) f12.shaderFloat16 = VK_TRUE;   // and the Float16 one the all-fp16 form declares
        f12.pNext = &cvf;
        f13.pNext = &f12;
    }
#endif

    VkDeviceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.pNext = &f13;
    ci.queueCreateInfoCount = 1; ci.pQueueCreateInfos = &qi;
    ci.pEnabledFeatures = &want;
    ci.enabledExtensionCount = (uint32_t)extensions.size();
    ci.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
    VK_CHECK(vkCreateDevice(g_phys, &ci, nullptr, &g_device));
    vkGetDeviceQueue(g_device, g_queue_family, 0, &g_queue);

    // The one device entry point stage 4 needs, fetched rather than linked because the loader exports no extension
    // function statically. Two such fetches in the whole program is the reason this tree declined a meta-loader.
#if NNTC_HAVE_COOPVEC
    if (g_cv_available) {
        g_cv_convert = (PFN_vkConvertCooperativeVectorMatrixNV)
            vkGetDeviceProcAddr(g_device, "vkConvertCooperativeVectorMatrixNV");
        if (!g_cv_convert) {
            fprintf(stderr, "ERROR: the device offers %s and has no vkConvertCooperativeVectorMatrixNV\n",
                    VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME);
            exit(1);
        }
    }
#endif
}

static void create_command_objects(void) {
    VkCommandPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = g_queue_family;
    VK_CHECK(vkCreateCommandPool(g_device, &pi, nullptr, &g_command_pool));

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = g_command_pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(g_device, &ai, &g_command_buffer));
}

// One image-layout change, as synchronisation2 states it: the stage and the access that reach it, in one struct.
static void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                          VkImageLayout from, VkImageLayout to,
                          VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                          VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkImageMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask = src_stage; b.srcAccessMask = src_access;
    b.dstStageMask = dst_stage; b.dstAccessMask = dst_access;
    b.oldLayout = from; b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = { aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS };
    VkDependencyInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
}

// One transient command buffer, submitted and waited on. Every load-time copy goes through this: a viewer's whole
// upload is a few megabytes at start-up, so a transfer queue and an asynchronous path would be ceremony with no
// reader and no gain.
static VkCommandBuffer begin_one_shot(void) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = g_command_pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(g_device, &ai, &cmd));
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    return cmd;
}

static void end_one_shot(VkCommandBuffer cmd) {
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkCommandBufferSubmitInfo csi{};
    csi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    csi.commandBuffer = cmd;
    VkSubmitInfo2 si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si.commandBufferInfoCount = 1; si.pCommandBufferInfos = &csi;
    VK_CHECK(vkQueueSubmit2(g_queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(g_queue));
    vkFreeCommandBuffers(g_device, g_command_pool, 1, &cmd);
}

// A buffer and its memory, which is the same four calls every time. `prefer`, when it is given, is a stricter set of
// properties to take if the device has a memory type with them, falling back to `props`; every caller but the
// frame readback passes nothing and gets exactly what it asks for.
static void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                          VkBuffer& buffer, VkDeviceMemory& memory, VkMemoryPropertyFlags prefer = 0) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(g_device, &bi, nullptr, &buffer));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g_device, buffer, &req);
    VkMemoryAllocateInfo ma{};
    ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ma.allocationSize = req.size;
    const uint32_t preferred = prefer ? find_memory_type_or_none(req.memoryTypeBits, prefer) : UINT32_MAX;
    ma.memoryTypeIndex = preferred != UINT32_MAX ? preferred : find_memory_type(req.memoryTypeBits, props);
    VK_CHECK(vkAllocateMemory(g_device, &ma, nullptr, &memory));
    VK_CHECK(vkBindBufferMemory(g_device, buffer, memory, 0));
}

// The depth image, at whatever the target's size is now. It is recreated with the swapchain, which is why it is torn
// down and built again rather than resized.
static void create_depth(uint32_t w, uint32_t h) {
    VkImageCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D; ii.format = g_depth_format;
    ii.extent = { w, h, 1 }; ii.mipLevels = 1; ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE; ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(g_device, &ii, nullptr, &g_depth_image));

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g_device, g_depth_image, &req);
    VkMemoryAllocateInfo ma{};
    ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ma.allocationSize = req.size;
    ma.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(g_device, &ma, nullptr, &g_depth_memory));
    VK_CHECK(vkBindImageMemory(g_device, g_depth_image, g_depth_memory, 0));

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = g_depth_image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = g_depth_format;
    vi.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    VK_CHECK(vkCreateImageView(g_device, &vi, nullptr, &g_depth_view));
}

static void destroy_depth(void) {
    if (g_depth_view) { vkDestroyImageView(g_device, g_depth_view, nullptr); g_depth_view = VK_NULL_HANDLE; }
    if (g_depth_image) { vkDestroyImage(g_device, g_depth_image, nullptr); g_depth_image = VK_NULL_HANDLE; }
    if (g_depth_memory) { vkFreeMemory(g_device, g_depth_memory, nullptr); g_depth_memory = VK_NULL_HANDLE; }
}

static void destroy_swapchain_objects(void) {
    for (VkSemaphore s : g_render_finished) vkDestroySemaphore(g_device, s, nullptr);
    g_render_finished.clear();
    for (VkImageView v : g_swapchain_views) vkDestroyImageView(g_device, v, nullptr);
    g_swapchain_views.clear();
    g_swapchain_images.clear();
    if (g_swapchain) { vkDestroySwapchainKHR(g_device, g_swapchain, nullptr); g_swapchain = VK_NULL_HANDLE; }
    destroy_depth();
}

// The swapchain, its views, its per-image semaphores and the depth image. Returns false when the window has no area at
// all - a minimised window - in which case the frame loop simply waits rather than creating a zero-sized swapchain.
static bool create_swapchain(void) {
    VkSurfaceCapabilitiesKHR caps{};
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_phys, g_surface, &caps));

    int cw = 0, ch = 0;
    win_client_size(&cw, &ch);
    VkExtent2D extent = caps.currentExtent;
    if (extent.width == 0xFFFFFFFFu) {   // the surface leaves the size to us, so the window's own client rect decides
        extent.width = (uint32_t)(cw > 0 ? cw : WINDOW_WIDTH);
        extent.height = (uint32_t)(ch > 0 ? ch : WINDOW_HEIGHT);
        if (extent.width < caps.minImageExtent.width) extent.width = caps.minImageExtent.width;
        if (extent.height < caps.minImageExtent.height) extent.height = caps.minImageExtent.height;
        if (extent.width > caps.maxImageExtent.width) extent.width = caps.maxImageExtent.width;
        if (extent.height > caps.maxImageExtent.height) extent.height = caps.maxImageExtent.height;
    }
    if (!extent.width || !extent.height) return false;

    uint32_t nfmt = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(g_phys, g_surface, &nfmt, nullptr));
    std::vector<VkSurfaceFormatKHR> formats(nfmt ? nfmt : 1);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(g_phys, g_surface, &nfmt, formats.data()));
    // Non-sRGB throughout, as in the Direct3D viewer: the latents are not colours and the decoder's output is written
    // as it is, so a format whose hardware applies a transfer function would change the picture. Taking formats[0]
    // when neither UNORM form is offered would do exactly that silently - the first entry is an sRGB one on several
    // drivers - so a surface that offers neither is a refusal by name and not a quietly wrong picture.
    const VkSurfaceFormatKHR* chosen = nullptr;
    for (const VkSurfaceFormatKHR& f : formats)
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { chosen = &f; break; }
    if (!chosen) {
        fprintf(stderr, "ERROR: this surface offers neither B8G8R8A8_UNORM nor R8G8B8A8_UNORM, and the decoder's "
                        "output must not go through a transfer function; it offers %u format(s):\n", nfmt);
        for (uint32_t i = 0; i < nfmt; i++)
            fprintf(stderr, "  VkFormat %d, colour space %d\n", (int)formats[i].format, (int)formats[i].colorSpace);
        exit(1);
    }
    g_swapchain_format = chosen->format;
    // Said once rather than on every recreate: a resize drags the window through dozens of them and the format is the
    // same every time, so only a change is worth a line.
    static VkFormat printed = VK_FORMAT_UNDEFINED;
    if (printed != g_swapchain_format) {
        printed = g_swapchain_format;
        printf("surface format: %s (VkFormat %d), colour space %d\n",
               g_swapchain_format == VK_FORMAT_B8G8R8A8_UNORM ? "B8G8R8A8_UNORM" : "R8G8B8A8_UNORM",
               (int)g_swapchain_format, (int)chosen->colorSpace);
    }

    uint32_t count = caps.minImageCount + 1;
    if (caps.maxImageCount && count > caps.maxImageCount) count = caps.maxImageCount;

    VkSwapchainCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = g_surface;
    ci.minImageCount = count;
    ci.imageFormat = chosen->format; ci.imageColorSpace = chosen->colorSpace;
    ci.imageExtent = extent; ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    // IDENTITY where the surface offers it, and currentTransform only where it does not. preTransform is a statement
    // about what the application has ALREADY applied: asking for IDENTITY leaves the rotation to the presentation
    // engine, which is right for a viewer that draws an unrotated scene, while passing currentTransform back on a
    // rotated display would promise a rotation this program's matrices never do and hand back a sideways picture.
    // (On a desktop compositor both are IDENTITY and the branch never fires; it is a phone and a tablet that rotate.)
    // Asking for a bit the surface does not list is a swapchain-creation failure rather than a fallback, hence the
    // test rather than an unconditional IDENTITY.
    ci.preTransform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
                    ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR : caps.currentTransform;
    // The first composite-alpha mode this surface actually supports, in the order this viewer would rather have them.
    // OPAQUE is what it wants and what Windows gives it, but the spec guarantees only that at least one bit is set:
    // on a compositor that owns the alpha channel OPAQUE can be absent, and creating a swapchain with an unsupported
    // bit is a failure rather than a fallback. INHERIT next, because it defers to whatever the window system already
    // decided; then the two pre-multiplied forms, which this scene's opaque alpha of 1.0 reads the same way under.
    static const VkCompositeAlphaFlagBitsKHR composite_order[] = {
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
    };
    VkCompositeAlphaFlagBitsKHR composite = (VkCompositeAlphaFlagBitsKHR)0;
    for (VkCompositeAlphaFlagBitsKHR bit : composite_order)
        if (caps.supportedCompositeAlpha & bit) { composite = bit; break; }
    if (!composite) {
        fprintf(stderr, "ERROR: this surface supports none of the four composite-alpha modes (0x%x)\n",
                (unsigned)caps.supportedCompositeAlpha);
        exit(1);
    }
    ci.compositeAlpha = composite;
    // FIFO is the one present mode Vulkan guarantees, and it is what the Direct3D viewer's Present(1, 0) does.
    ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    ci.clipped = VK_TRUE;
    VK_CHECK(vkCreateSwapchainKHR(g_device, &ci, nullptr, &g_swapchain));
    g_swapchain_extent = extent;

    uint32_t nimg = 0;
    VK_CHECK(vkGetSwapchainImagesKHR(g_device, g_swapchain, &nimg, nullptr));
    g_swapchain_images.resize(nimg);
    VK_CHECK(vkGetSwapchainImagesKHR(g_device, g_swapchain, &nimg, g_swapchain_images.data()));
    g_swapchain_views.resize(nimg);
    g_render_finished.resize(nimg);
    for (uint32_t i = 0; i < nimg; i++) {
        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = g_swapchain_images[i]; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = g_swapchain_format;
        vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(g_device, &vi, nullptr, &g_swapchain_views[i]));
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK(vkCreateSemaphore(g_device, &si, nullptr, &g_render_finished[i]));
    }
    create_depth(extent.width, extent.height);
    return true;
}

// ---------------------------------------------------------------------------
// 4x4 matrices, the Direct3D viewer's own, row-major with the column-vector convention and one transpose at
// buffer-write time. Copied rather than adapted: a camera that is to be compared parameter for parameter has to be
// built by the same arithmetic in the same order.
// ---------------------------------------------------------------------------
struct Mat4 { float m[16]; };
static Mat4 mat_identity() { Mat4 r{}; for (int i = 0; i < 4; i++) r.m[i * 4 + i] = 1.0f; return r; }
static Mat4 mat_mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { float s = 0; for (int k = 0; k < 4; k++) s += a.m[i * 4 + k] * b.m[k * 4 + j]; r.m[i * 4 + j] = s; }
    return r;
}
// Vulkan's depth range is 0 to 1, which is what this matrix already produces, so it is copied as it stands and the y
// flip is a negative viewport height rather than a sign here (the plan's section 1.8).
static Mat4 mat_perspective(float fov_deg, float aspect, float znear, float zfar) {
    Mat4 m{};
    const float f = 1.0f / std::tan((fov_deg * 3.14159265358979f / 180.0f) / 2.0f);
    m.m[0 * 4 + 0] = f / aspect;
    m.m[1 * 4 + 1] = f;
    m.m[2 * 4 + 2] = zfar / (znear - zfar);
    m.m[2 * 4 + 3] = (zfar * znear) / (znear - zfar);
    m.m[3 * 4 + 2] = -1.0f;
    return m;
}
static Mat4 mat_translate(float x, float y, float z) { Mat4 m = mat_identity(); m.m[0 * 4 + 3] = x; m.m[1 * 4 + 3] = y; m.m[2 * 4 + 3] = z; return m; }
static Mat4 mat_rot_y(float deg) {
    Mat4 m = mat_identity(); const float r = deg * 3.14159265358979f / 180.0f, c = std::cos(r), s = std::sin(r);
    m.m[0 * 4 + 0] = c; m.m[0 * 4 + 2] = s; m.m[2 * 4 + 0] = -s; m.m[2 * 4 + 2] = c; return m;
}
static Mat4 mat_rot_x(float deg) {
    Mat4 m = mat_identity(); const float r = deg * 3.14159265358979f / 180.0f, c = std::cos(r), s = std::sin(r);
    m.m[1 * 4 + 1] = c; m.m[1 * 4 + 2] = -s; m.m[2 * 4 + 1] = s; m.m[2 * 4 + 2] = c; return m;
}
static Mat4 mat_transpose(const Mat4& a) { Mat4 r{}; for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) r.m[i * 4 + j] = a.m[j * 4 + i]; return r; }

// ---------------------------------------------------------------------------
// The uniform block: the Direct3D viewer's two constant buffers laid end to end, and it has to be exactly that,
// because the packing code below is that viewer's copied. Every member is a vec4, an ivec4, a mat4 or an array of
// vec4, so std140's rules and the C struct's agree with no padding thought required - std140's array stride for a vec4
// is 16, which is the array's natural stride.
// ---------------------------------------------------------------------------
struct SceneConstants {      // the Direct3D viewer's CBData (b0)
    float mvp[16];
    float tex_size[4];
    float lod_info[4];
    float const0[4];
    float const1[4];
};
struct DecoderConstants {    // the Direct3D viewer's DecCB (b1)
    float lo0[4], hi0[4], lo1[4], hi1[4];
    int   dims[4];           // C0, C1, nin, nout
    int   sel[4];            // the terms mask, the output texture shown, level 0's texture count, 0
    float W[108][4];         // row r, column c at W[r * 6 + c / 4][c % 4]
    float bias[5][4];        // nout <= 18, so five vec4 cover it
};
struct Uniforms { SceneConstants scene; DecoderConstants dec; };
static_assert(sizeof(SceneConstants) == 128, "the scene block is the other viewer's 128 bytes");
static_assert(sizeof(DecoderConstants) == 1904, "the decoder block is the other viewer's 1904 bytes");
static_assert(sizeof(Uniforms) == 2032, "the uniform buffer is the two blocks end to end");

// ---------------------------------------------------------------------------
// The asset
// ---------------------------------------------------------------------------
struct GpuImage {
    VkImage        image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView    view = VK_NULL_HANDLE;
};

// The three objects of one texture, in the order a view must go before what it views. Every handle is checked and
// cleared, so calling this on an image that was never created, or twice, does nothing.
static void destroy_image(GpuImage& img) {
    if (img.view) { vkDestroyImageView(g_device, img.view, nullptr); img.view = VK_NULL_HANDLE; }
    if (img.image) { vkDestroyImage(g_device, img.image, nullptr); img.image = VK_NULL_HANDLE; }
    if (img.memory) { vkFreeMemory(g_device, img.memory, nullptr); img.memory = VK_NULL_HANDLE; }
}

struct Latent {
    GpuImage img;            // the level's texture, or the FIRST file of a two-file level 0
    GpuImage img_b;          // the second file (channels 2-3 as a BC5, or channel 2 alone as a BC4)
    // Level 0's load-time BC4 / BC5 pack (key 4), which exists only when the file's own level 0 is UNCOMPRESSED: the
    // packed texture count, each one's channel count, the packing quality over the chain and the round trip's largest
    // error, exactly as the Direct3D viewer reports them, because it is the same shared/bc_pack.h doing the packing.
    GpuImage bc_img[2];
    int bc_n = 0, bc_nc[2] = { 0, 0 };
    double bc_psnr = 0.0, bc_maxerr = 0.0;
    bool bc_lossless = false;
    std::vector<std::vector<uint8_t>> raw;          // the uncompressed .dds levels as read, kept only until the pack has run
    std::vector<std::pair<int, int>> dims;          // every level's extent, which the overlay's memory figure walks
    int W = 0, H = 0, C = 0, stored_C = 0, mips = 0, dxgi = 0;
    // The SECOND file's own format and channel count, which are not the first's: a three-channel level 0 is a BC5 of
    // channels 0-1 and a BC4 of channel 2 alone, so the two files differ in both. Zero when there is only one file.
    int stored_C_b = 0, dxgi_b = 0;
    int files = 1;           // the .dds files this level is stored in
    bool file_bc = false;    // the level came block-compressed in the file
    float lo[4] = { 0, 0, 0, 0 }, hi[4] = { 1, 1, 1, 1 };
    int bits[4] = { 8, 8, 8, 8 };
};

struct State {
    float x = 0.0f, y = 0.0f, z = -3.0f, yaw = 0.0f, pitch = 0.0f;
    bool  cube = false;
    int   filter_mode = 2;   // 0 point, 1 bilinear, 2 trilinear (the default, as in the other viewer)
    bool  aniso = true;      // key X: anisotropy, which applies in TRILINEAR mode only
    bool  lod_bias = true;   // key L: level 1's UV gradients scaled by 2^lod_bias_level1 (the encoder's 1:1 mip rule)
    bool  mips_on = true;    // key M: false = the sampler's maxLod is 0, so every fetch reads mip 0
    Latent lat[2];
    Uniforms cb{};
    int   textures_out = 1, tex_shown = 0;
    int   src_w = 0, src_h = 0, block = 4;
    float lod_bias_level1 = 2.0f;   // the descriptor's own value: the mip levels level 1 is shifted by, so the gradient scale is 2^this
    std::string terms;
    bool  debug_dirty = true;   // the overlay's text is rebuilt and re-uploaded only when something it says has changed
};
static State g;

// Key 4: level 0 bound from the BC4 / BC5 pack made at load rather than from the file's own (uncompressed) plane.
// --bc starts there. On a file whose level 0 is already block-compressed there is no pack and the key does nothing,
// which is the other viewer's behaviour and not an accident of this one.
static bool g_bc = false;
// --nooverlay: the scene alone, so that two --shot frames can be compared byte for byte. The strip names the file and
// its format, which two renders that must be identical would still differ inside.
static bool g_overlay = true;

// The camera's speeds, the other viewer's constants: one line each and the same units, so a --z on one command line
// means what it means on the other's.
static const float Z_MIN = 0.40f, Z_MAX = -50.0f;
static const float Z_SPEED = 1.0f, XY_SPEED = 0.75f, ROT_SPEED = 90.0f;

static const char* filter_name(int m) { return m == 0 ? "POINT" : (m == 2 ? "TRILINEAR" : "BILINEAR"); }

// The sampler slot a filter state picks: anisotropy is the trilinear filter with several samples along the
// footprint's long axis, so it is a fourth state of the trilinear mode and not a mode of its own. In point and
// bilinear the flag is remembered and does nothing, which is what the other viewer's overlay says.
static int filter_slot(void) { return (g.filter_mode == 2 && g.aniso) ? 3 : g.filter_mode; }

// The eight samplers of the plan's section 1.10: [mips on / off][point, bilinear, trilinear, anisotropic]. There used
// to be sixteen, a second set carrying level 1's mipLodBias; level 1's LOD shift is in the gradients now (const0.z),
// so BOTH levels sample through this one table and no sampler here carries a bias at all.
static VkSampler g_samplers[2][FILTER_STATES] = {};

static VkSampler make_sampler(int filter, bool mips) {
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = filter == 0 ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    si.minFilter = si.magFilter;
    si.mipmapMode = filter >= 2 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    // CLAMP throughout, because the encoder fits the taps clamped at the edges.
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.mipLodBias = 0.0f;   // no sampler in this viewer carries a LOD bias; see create_samplers
    si.anisotropyEnable = (filter == 3 && g_has_aniso) ? VK_TRUE : VK_FALSE;
    si.maxAnisotropy = si.anisotropyEnable ? ANISO_MAX : 1.0f;
    si.compareEnable = VK_FALSE;
    si.compareOp = VK_COMPARE_OP_NEVER;
    si.minLod = 0.0f;
    // maxLod 0 is the other viewer's MaxLOD = 0 set (key M): the hardware clamps every fetch to mip 0 with nothing
    // reloaded and no texture touched.
    si.maxLod = mips ? VK_LOD_CLAMP_NONE : 0.0f;
    si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    si.unnormalizedCoordinates = VK_FALSE;
    VkSampler s = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSampler(g_device, &si, nullptr, &s));
    return s;
}

// The encoder's 1:1 mip rule on the GPU: output mip m reads mip m of BOTH latents. Level 1 has a quarter of the texels
// per axis, so its natural LOD is log2(block) = 2 below level 0's, and the descriptor's lod_bias_level1 is how many
// mips it must be shifted up by. That shift is applied IN THE GRADIENTS - view.frag's textureGrad multiplies both UV
// derivatives by 2^lod_bias_level1, which raises the LOD the hardware computes by exactly that many mips before any
// clamp - and NOT as a sampler mipLodBias, which is the other viewer's choice too and for the same reason: a sampler
// bias is equivalent only where the hardware keeps a properly negative base LOD under magnification, and on an Intel
// Xe integrated GPU it does not (base floored near 0, plus 2, mip 2, a badly blurred magnified picture, in this
// viewer and in the Direct3D one alike). Intel's PRM documents bias-then-clamp with an implementation-dependent base.
//
// The value is part of the format's contract (docs/FORMAT.md section 5), so it is read from the descriptor rather
// than assumed - and a number outside the range the shift can mean is refused by name, before a sampler exists,
// rather than turned into a gradient scale of 2^1000. (The refusal this replaces was
// against maxSamplerLodBias, which no longer constrains anything here: no sampler carries a bias.)
static const float LOD_BIAS_LEVEL1_MAX = 8.0f;   // a block of 256: far past anything the format writes, and still a finite scale
static bool check_level1_lod_shift(float bias) {
    if (!(bias >= 0.0f && bias <= LOD_BIAS_LEVEL1_MAX)) {
        fprintf(stderr, "ERROR: the descriptor's lod_bias_level1 is %g, which is outside [0, %g]: it is the number of mip\n"
                        "       levels level 1 is shifted by, so it cannot be negative and cannot be that large\n",
                (double)bias, (double)LOD_BIAS_LEVEL1_MAX);
        return false;
    }
    printf("  level 1 LOD: UV gradients scaled by %g (2^lod_bias_level1, lod_bias_level1 = %g%s); key L toggles it\n",
           (double)std::exp2(bias), (double)bias,
           bias == std::log2((float)g.block) ? ", log2 of the block" : ", NOT log2 of the block");
    return true;
}

static void create_samplers(void) {
    for (int mips = 0; mips < 2; mips++) for (int f = 0; f < FILTER_STATES; f++)
        g_samplers[mips][f] = make_sampler(f, mips != 0);
}

static VkFormat vk_format_for_dxgi(int dxgi) {
    switch (dxgi) {
        case 61: return VK_FORMAT_R8_UNORM;
        case 49: return VK_FORMAT_R8G8_UNORM;
        case 28: return VK_FORMAT_R8G8B8A8_UNORM;
        case 80: return VK_FORMAT_BC4_UNORM_BLOCK;
        case 83: return VK_FORMAT_BC5_UNORM_BLOCK;
        default: return VK_FORMAT_UNDEFINED;
    }
}

// The five formats above by their spec names, so that a refusal says BC5_UNORM_BLOCK rather than 131 and the reader
// can look the name up without a header open beside them.
static const char* vk_format_name(VkFormat fmt) {
    switch (fmt) {
        case VK_FORMAT_R8_UNORM: return "R8_UNORM";
        case VK_FORMAT_R8G8_UNORM: return "R8G8_UNORM";
        case VK_FORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
        case VK_FORMAT_BC4_UNORM_BLOCK: return "BC4_UNORM_BLOCK";
        case VK_FORMAT_BC5_UNORM_BLOCK: return "BC5_UNORM_BLOCK";
        default: return "an unnamed format";
    }
}

// One level of an image about to be uploaded: where its bytes are, how many there are and what its extent is.
struct UploadLevel { const uint8_t* data; size_t size; int w; int h; };

// Every level of one texture into one image, through one staging buffer. Both callers go through it: the .dds files
// the descriptor names, and the BC4 / BC5 pack this viewer makes of an uncompressed level 0 at load.
//
// The per-level offsets in the staging buffer are 16-BYTE ALIGNED rather than the source's own tightly packed ones,
// because VkBufferImageCopy::bufferOffset must be a multiple of 4 and of the format's texel-block size: a BC5 chain is
// fine as it lies in the file, but an R8 chain with an odd width is not. Re-laying the levels out costs one memcpy per
// level and removes the whole class of problem.
static bool upload_image(const std::vector<UploadLevel>& levels, int W, int H, VkFormat fmt, const std::string& what,
                         GpuImage& out) {
    // The four lines that turn "the picture is wrong" into "this device cannot sample this format". Linear filtering
    // is not optional here: the asset was fitted under a filtered sample, so a format without it is a refusal. This is
    // the BACKSTOP and not the first line of defence: a BC asset on a device with no textureCompressionBC is refused
    // from the descriptor before a file is opened (refuse_bc_asset_without_bc), and what reaches here is a format the
    // descriptor did not publish, or one this device declines for some reason of its own.
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(g_phys, fmt, &fp);
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    if ((fp.optimalTilingFeatures & need) != need) {
        const bool bc = fmt == VK_FORMAT_BC4_UNORM_BLOCK || fmt == VK_FORMAT_BC5_UNORM_BLOCK;
        fprintf(stderr, "ERROR: '%s': this device cannot sample %s (VkFormat %d) with linear filtering%s\n",
                what.c_str(), vk_format_name(fmt), (int)fmt,
                bc && !g_has_bc ? " (it does not report textureCompressionBC)" : "");
        return false;
    }
    const int nmip = (int)levels.size();

    VkImageCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D; ii.format = fmt;
    ii.extent = { (uint32_t)W, (uint32_t)H, 1 };
    ii.mipLevels = (uint32_t)nmip; ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE; ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(g_device, &ii, nullptr, &out.image));
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g_device, out.image, &req);
    VkMemoryAllocateInfo ma{};
    ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ma.allocationSize = req.size;
    ma.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(g_device, &ma, nullptr, &out.memory));
    VK_CHECK(vkBindImageMemory(g_device, out.image, out.memory, 0));

    std::vector<VkBufferImageCopy> regions((size_t)nmip);
    VkDeviceSize total = 0;
    for (int i = 0; i < nmip; i++) {
        regions[(size_t)i].bufferOffset = total;
        regions[(size_t)i].bufferRowLength = 0;     // tightly packed, which each level is
        regions[(size_t)i].bufferImageHeight = 0;
        regions[(size_t)i].imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t)i, 0, 1 };
        regions[(size_t)i].imageOffset = { 0, 0, 0 };
        // For a block format the extent is in TEXELS, not blocks, and a final level smaller than 4x4 is legal
        // precisely because it is the whole mip.
        regions[(size_t)i].imageExtent = { (uint32_t)levels[(size_t)i].w, (uint32_t)levels[(size_t)i].h, 1 };
        total += (VkDeviceSize)((levels[(size_t)i].size + 15) & ~(size_t)15);
    }

    VkBuffer staging = VK_NULL_HANDLE; VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    create_buffer(total, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, staging_memory);
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(g_device, staging_memory, 0, total, 0, &mapped));
    for (int i = 0; i < nmip; i++)
        memcpy((uint8_t*)mapped + regions[(size_t)i].bufferOffset, levels[(size_t)i].data, levels[(size_t)i].size);
    vkUnmapMemory(g_device, staging_memory);

    VkCommandBuffer cmd = begin_one_shot();
    image_barrier(cmd, out.image, VK_IMAGE_ASPECT_COLOR_BIT,
                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    vkCmdCopyBufferToImage(cmd, staging, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           (uint32_t)regions.size(), regions.data());
    image_barrier(cmd, out.image, VK_IMAGE_ASPECT_COLOR_BIT,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    end_one_shot(cmd);
    vkDestroyBuffer(g_device, staging, nullptr);
    vkFreeMemory(g_device, staging_memory, nullptr);

    // Identity swizzle: the shader reads only .r of the second texture when C0 is 3, exactly as the Direct3D shader
    // does, so no channel that is not there is ever part of phi and nothing has to imitate Direct3D's BC4 (r, 0, 0, 1).
    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = out.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, (uint32_t)nmip, 0, 1 };
    VK_CHECK(vkCreateImageView(g_device, &vi, nullptr, &out.view));
    return true;
}

// One .dds file's levels, as the loader read them, into one image at the format the DXGI id maps to.
static bool upload_dds(const DdsImage& img, const std::string& path, GpuImage& out) {
    const VkFormat fmt = vk_format_for_dxgi(img.dxgi);
    if (fmt == VK_FORMAT_UNDEFINED) {
        fprintf(stderr, "ERROR: '%s': DXGI format %d has no Vulkan equivalent this viewer samples\n", path.c_str(), img.dxgi);
        return false;
    }
    std::vector<UploadLevel> levels((size_t)img.mips);
    for (int i = 0; i < img.mips; i++)
        levels[(size_t)i] = { (const uint8_t*)img.bytes.data() + img.levels[(size_t)i].offset,
                              img.levels[(size_t)i].size, img.levels[(size_t)i].w, img.levels[(size_t)i].h };
    return upload_image(levels, img.W, img.H, fmt, path, out);
}

// The .dds loader's Vulkan tail. The parsing is shared/dds.h - the same header walk, the same bounds and the same
// "bytes expected / bytes present" check the Direct3D viewer makes - so the two viewers refuse the same files for the
// same reasons. `second` loads the SECOND file of a two-file level 0.
static bool load_dds(const std::string& path, Latent& L, bool second = false) {
    DdsImage img;
    if (!dds_read(path, img)) return false;
    // The two files of one level share the extent and the level count, and NOT the format: a three-channel level 0 is
    // a BC5 of channels 0-1 and a BC4 of channel 2, so only that both are block-compressed (or neither is) is required.
    if (second && (img.W != L.W || img.H != L.H || img.mips != L.mips)) {
        fprintf(stderr, "ERROR: '%s' does not match the first file of this level (%dx%d, %d level%s)\n",
                path.c_str(), L.W, L.H, L.mips, L.mips == 1 ? "" : "s");
        return false;
    }
    if (second && img.bc != L.file_bc) {
        fprintf(stderr, "ERROR: '%s' is %s where the first file of this level is not\n", path.c_str(),
                img.bc ? "block-compressed" : "uncompressed");
        return false;
    }
    // THE BASE OF A BLOCK-COMPRESSED TEXTURE MUST BE A MULTIPLE OF 4, and ONLY the base: this tree supports
    // non-power-of-two and non-square textures, but a block-compressed base is divisible by 4 texels on each axis. The
    // LOWER levels are not checked and must not be - a 360x200 base gives a perfectly ordinary 90x50 level, and
    // examples/npot360x200 is in the tree precisely because three of its five levels are off the block grid.
    //
    // Measured on the driver this was written against: vkCreateImage takes such a base, the validation layer says
    // nothing, and the asset is drawn as though its base were the size the encoder wrote - a picture and no error,
    // which is what the WebGPU viewer was caught doing. That is one driver and not a guarantee; Direct3D 12 makes the
    // same thing an explicit capability (OPTIONS8's UnalignedBlockTexturesSupported), so a base off the block grid is
    // something an implementation MAY take, not something any of them must. Which is why it is refused here rather
    // than left to the driver: an asset that draws on one machine and fails on another is worse than one refused on
    // both, and this tree does not support such a base. The refusal is made before the image exists, where it can say
    // which file and how big it is. nntc_encode pads the base, so an asset out of this tree cannot reach this line; a
    // hand-made or third-party .dds can.
    if (img.bc && ((img.W % 4) || (img.H % 4))) {
        fprintf(stderr, "ERROR: '%s' is %s and its base is %dx%d; a block-compressed base must be a multiple of 4 on\n"
                        "       both axes (the lower mip levels need not be). nntc_encode pads the base, so this file\n"
                        "       was not written by it.\n",
                path.c_str(), img.dxgi == 80 ? "BC4_UNORM" : "BC5_UNORM", img.W, img.H);
        return false;
    }
    if (!upload_dds(img, path, second ? L.img_b : L.img)) return false;
    if (second) { L.stored_C_b = img.channels; L.dxgi_b = img.dxgi; }
    else {
        L.W = img.W; L.H = img.H; L.stored_C = img.channels; L.mips = img.mips; L.dxgi = img.dxgi; L.file_bc = img.bc;
        L.dims.clear();
        for (int i = 0; i < img.mips; i++) L.dims.push_back({ img.levels[(size_t)i].w, img.levels[(size_t)i].h });
        // The uncompressed levels are kept only so that the load-time pack can read them; it drops them once it has
        // run, because a level-0 chain is the largest thing this program holds.
        L.raw.clear();
        if (!img.bc) {
            L.raw.resize((size_t)img.mips);
            for (int i = 0; i < img.mips; i++)
                L.raw[(size_t)i].assign((const uint8_t*)img.bytes.data() + img.levels[(size_t)i].offset,
                                        (const uint8_t*)img.bytes.data() + img.levels[(size_t)i].offset + img.levels[(size_t)i].size);
        }
    }
    printf("  %s: %dx%d, %d mip level%s, DXGI format %d (%s, %d channel%s stored)\n", path.c_str(), img.W, img.H,
           img.mips, img.mips == 1 ? "" : "s", img.dxgi,
           img.bc ? (img.dxgi == 80 ? "BC4_UNORM" : "BC5_UNORM") : "uncompressed", img.channels,
           img.channels == 1 ? "" : "s");
    return true;
}

// Level 0's packed textures from its .dds levels, in bc_pack.h's own file rule: channels 0-1 into a BC5 (a BC4 when
// that is all there is), and what is left into a second texture - a BC5 for a fourth channel, a BC4 for a third one
// alone. It is the same split the encoder writes, so key 4 shows the layout the file format would have given the same
// plane, and it is the same header doing the packing as in the Direct3D viewer, so the two produce the same blocks and
// print the same numbers.
//
// This exists for an UNCOMPRESSED level 0 only. Under the encoder's default level 0 already arrives block-compressed
// and optimised as BC, and there is nothing here to improve on: the caller does not call this, key 4 has nothing to
// bind and does nothing, which is what the other viewer does too.
// The pack gave up part way through - the second texture failed where the first was already created and bound to
// memory. Everything it made is destroyed and every field it set is reset, so that key 4 has nothing to bind rather
// than half a plane, and so that the images it did create are not leaked for the life of the program.
static void abandon_bc_pack(Latent& L) {
    for (int p = 0; p < 2; p++) { destroy_image(L.bc_img[p]); L.bc_nc[p] = 0; }
    L.bc_n = 0;
    L.bc_psnr = 0.0; L.bc_maxerr = 0.0; L.bc_lossless = false;
}

static bool bc_pack_level0(Latent& L) {
    const int C = L.C;
    L.bc_n = bc_level0_files(C);
    if ((L.W % 4) || (L.H % 4)) {
        fprintf(stderr, "WARNING: level 0 is %dx%d, not a multiple of 4: no BC pack\n", L.W, L.H);
        abandon_bc_pack(L);
        return false;
    }
    double se_all = 0; size_t n_all = 0; bool lossless = true;
    printf("  level 0 packed at load (key 4 switches to it): ");
    for (int p = 0; p < L.bc_n; p++) {
        const int c0 = 2 * p, nc = bc_level0_file_channels(C, p);
        L.bc_nc[p] = nc;
        if (c0 + nc > L.stored_C) {
            fprintf(stderr, "\nERROR: level 0 stores %d channels, %d needed for the pack\n", L.stored_C, c0 + nc);
            abandon_bc_pack(L);
            return false;
        }
        const std::string second_channel = nc == 1 ? std::string() : " and " + std::to_string(c0 + 1);
        printf("%s%s (channel%s %d%s, %d bit%s)", p ? " + " : "", nc == 1 ? "BC4" : "BC5", nc == 1 ? "" : "s", c0,
               second_channel.c_str(), L.bits[c0], L.bits[c0] == 1 ? "" : "s");
        for (int c = 0; c < nc; c++) if (L.bits[c0 + c] > 3) lossless = false;   // every packed channel is one the layout uses
        std::vector<std::vector<uint8_t>> packed((size_t)L.mips);
        std::vector<UploadLevel> levels((size_t)L.mips);
        for (int i = 0; i < L.mips; i++) {
            double se = 0; size_t n = 0;
            const int w = L.dims[(size_t)i].first, h = L.dims[(size_t)i].second;
            packed[(size_t)i] = bc_pack_level(L.raw[(size_t)i].data(), w, h, L.stored_C, c0, nc, L.bits, C, se, n);
            {   // The round trip: every packed block decoded on the CPU (the hardware palette) against the exact index
                // value of the source. One decode per BLOCK and then its sixteen texels, rather than a decode per texel.
                const int bx = (w + 3) / 4, by = (h + 3) / 4;
                for (int byi = 0; byi < by; byi++) for (int bxi = 0; bxi < bx; bxi++) for (int c = 0; c < nc; c++) {
                    float dec[16];
                    bc4_decode_block(&packed[(size_t)i][((size_t)byi * bx + bxi) * (nc == 1 ? 8 : 16) + (size_t)c * 8], dec);
                    const int bits_c = L.bits[c0 + c];
                    for (int t = 0; t < 16; t++) {
                        const int x = bxi * 4 + t % 4, y = byi * 4 + t / 4;
                        if (x >= w || y >= h) continue;
                        const int k = L.raw[(size_t)i][((size_t)y * w + x) * L.stored_C + c0 + c] >> (8 - bits_c);
                        // In DOUBLE, as viewer/main.cpp does it: dec[t] is promoted and the exact index value stays a
                        // double, so the two viewers print the same "largest error" for the same asset. Rounding the
                        // right-hand side to float first made this viewer's number differ in the last digits.
                        L.bc_maxerr = std::max(L.bc_maxerr, std::fabs((double)dec[t] - k * 255.0 / ((1 << bits_c) - 1)));
                    }
                }
            }
            levels[(size_t)i] = { packed[(size_t)i].data(), packed[(size_t)i].size(), w, h };
            se_all += se; n_all += n;
        }
        if (!upload_image(levels, L.W, L.H, nc == 1 ? VK_FORMAT_BC4_UNORM_BLOCK : VK_FORMAT_BC5_UNORM_BLOCK,
                          "the level-0 pack made at load", L.bc_img[p])) {
            fprintf(stderr, "\nERROR: the BC texture creation failed\n");
            abandon_bc_pack(L);
            return false;
        }
    }
    L.bc_lossless = lossless;
    L.bc_psnr = n_all && se_all > 0 ? 10.0 * std::log10(255.0 * 255.0 / (se_all / (double)n_all)) : 0.0;
    if (lossless)
        printf(": lossless (0 and 255 as the endpoints, the index in the selectors; the round trip's largest error "
               "%.3g / 255 in float)\n", L.bc_maxerr);
    else
        printf(": lossy, packing PSNR %.2f dB over the chain against the exact index values (largest error %.1f / 255)\n",
               L.bc_psnr, L.bc_maxerr);
    return true;
}

// The 1x1 image the unused third slot is given. Vulkan does not allow a null image view in a descriptor without
// VK_EXT_robustness2's nullDescriptor, where the Direct3D viewer simply binds a null shader-resource view to t2. It is
// never sampled - sel.z tells the shader how many textures level 0 is read from, as it does there - but it must be a
// valid view, and its contents are therefore never read and never written.
static GpuImage g_dummy;

// --shot's offscreen colour target, file-scope for the reason render_shot gives: every path out of that function,
// including the one a shader that will not compile takes, must leave nothing behind for vkDestroyDevice to find.
static GpuImage g_shot_target;
static void create_dummy_image(void) {
    VkImageCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D; ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent = { 1, 1, 1 }; ii.mipLevels = 1; ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE; ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(g_device, &ii, nullptr, &g_dummy.image));
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g_device, g_dummy.image, &req);
    VkMemoryAllocateInfo ma{};
    ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ma.allocationSize = req.size;
    ma.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(g_device, &ma, nullptr, &g_dummy.memory));
    VK_CHECK(vkBindImageMemory(g_device, g_dummy.image, g_dummy.memory, 0));
    VkCommandBuffer cmd = begin_one_shot();
    image_barrier(cmd, g_dummy.image, VK_IMAGE_ASPECT_COLOR_BIT,
                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    end_one_shot(cmd);
    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = g_dummy.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = VK_FORMAT_R8G8B8A8_UNORM;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VK_CHECK(vkCreateImageView(g_device, &vi, nullptr, &g_dummy.view));
}

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::string();
    std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}

// ---------------------------------------------------------------------------
// The asset: PREFIX_nntc.json, then the .dds files it names (next to it), then the decoder constants. Every refusal
// below is the Direct3D viewer's, word for word where the words are the same, because the two viewers must accept and
// refuse exactly the same files - a descriptor one of them opens and the other does not would make the whole
// comparison meaningless.
// ---------------------------------------------------------------------------

// A block-compressed asset on a device that does not report textureCompressionBC, refused ONCE and from the
// DESCRIPTOR, before a single .dds is opened.
//
// The backstop in upload_image stays and still names the format, but arriving there is the wrong shape of failure: it
// comes after the file reads, it comes once per file, and on the load-time pack it comes as a stack of ERROR lines in
// a run that then carries on. The descriptor already publishes each texture's dxgi_format_id, so the one thing the
// user needs to know is knowable before anything is opened - and the line says what to do about it rather than only
// what is wrong, because the encoder can write this same material in a form this device does sample.
//
// 80 and 83 are BC4_UNORM and BC5_UNORM; the older bare-string spelling of a "files" entry publishes no id, in which
// case there is nothing here to read and upload_image's check is what refuses.
static bool refuse_bc_asset_without_bc(const JVal& root, const std::string& json_path) {
    if (g_has_bc) return true;
    const JVal* texs = root.get("textures");
    if (!texs || texs->kind != JVal::ARR) return true;   // a malformed descriptor is load_asset's own refusal, below
    const char* bc_name = nullptr;
    const auto note = [&bc_name](int dxgi) {
        // BC5 is named in preference to BC4 when both are there: it is the wider of the two and the one a two-file
        // level 0 leads with, so it is the format the reader will recognise the asset by.
        if (dxgi == 83) bc_name = "BC5_UNORM_BLOCK";
        else if (dxgi == 80 && !bc_name) bc_name = "BC4_UNORM_BLOCK";
    };
    for (const JVal& t : texs->arr) {
        note((int)t.number("dxgi_format_id", 0));
        const JVal* files = t.get("files");
        if (files && files->kind == JVal::ARR)
            for (const JVal& e : files->arr) if (e.kind == JVal::OBJ) note((int)e.number("dxgi_format_id", 0));
    }
    if (!bc_name) return true;
    fprintf(stderr, "ERROR: '%s' stores its latents as %s and this device does not report textureCompressionBC, so it "
                    "cannot sample any BC format; encode with --l0 palette --bits0 N --bc0 0 (or --bc0 both) for an "
                    "uncompressed level 0 this device can sample\n", json_path.c_str(), bc_name);
    return false;
}

static bool load_asset(const std::string& json_path) {
    const std::string text = read_file(json_path);
    if (text.empty()) { fprintf(stderr, "ERROR: cannot read '%s'\n", json_path.c_str()); return false; }
    JParser P(text); JVal root = P.parse();
    if (!P.ok || root.kind != JVal::OBJ) { fprintf(stderr, "ERROR: '%s' is not valid JSON\n", json_path.c_str()); return false; }
    // Assets written before the project was renamed carry the older format string; they are the same format byte for
    // byte, so the viewer opens them and says so once rather than refusing what it can read.
    if (root.string("format") == "ntc-dds-1") printf("note: format 'ntc-dds-1' is the older name of nntc-dds-1; reading it as such\n");
    else if (root.string("format") != "nntc-dds-1") { fprintf(stderr, "ERROR: unknown format '%s' (expected nntc-dds-1)\n", root.string("format").c_str()); return false; }
    printf("Loading %s\n", json_path.c_str());
    if (!refuse_bc_asset_without_bc(root, json_path)) return false;
    const size_t slash = json_path.find_last_of("/\\");
    const std::string dir = slash == std::string::npos ? "" : json_path.substr(0, slash + 1);
    // A file the descriptor names is taken as written when it is an absolute path, and from the descriptor's own
    // directory otherwise: the encoder writes base names, and a hand-authored descriptor may point anywhere.
    const auto beside = [&dir](const std::string& name) {
        const std::filesystem::path p(name);
        return p.is_absolute() ? name : dir + name;
    };
    const JVal* src = root.get("source"); const JVal* decs = root.get("decode");
    g.src_w = src ? (int)src->number("width") : 0; g.src_h = src ? (int)src->number("height") : 0;
    g.textures_out = decs ? (int)decs->number("textures_out", 1) : 1; g.block = (int)root.number("block", 4);
    const JVal* texs = root.get("textures");
    if (!texs || texs->kind != JVal::ARR || texs->arr.size() != 2) { fprintf(stderr, "ERROR: the JSON must describe two textures\n"); return false; }
    for (int l = 0; l < 2; l++) {
        const JVal& t = texs->arr[(size_t)l]; Latent& L = g.lat[l];
        if ((int)t.number("level", -1) != l) { fprintf(stderr, "ERROR: textures[%d] is not level %d\n", l, l); return false; }
        L.C = (int)t.number("channels_used");
        // bits_per_channel decides a shift and a divisor in every reader of this format, so 0 or a value above 8 would
        // be a negative shift or a division by zero. The entries are checked for being numbers at all, too: a JSON
        // that carried a string there would otherwise read as .num = 0.
        const JVal* bits = t.get("bits_per_channel");
        for (int c = 0; c < 4; c++) L.bits[c] = bits && bits->kind == JVal::ARR && c < (int)bits->arr.size() && bits->arr[(size_t)c].kind == JVal::NUM ? (int)bits->arr[(size_t)c].num : 8;
        for (int c = 0; c < 4; c++) {
            const int lo_bits = l == 0 ? 1 : 4;   // level 0 is written at 8 bits under --l0 bc8 and at 1-4 under --l0 palette; level 1 at 4-8
            if (L.bits[c] < lo_bits || L.bits[c] > 8) { fprintf(stderr, "ERROR: level %d: bits_per_channel[%d] is %d, which is outside %d..8\n", l, c, L.bits[c], lo_bits); return false; }
        }
        // "file" names one .dds and "files" names two, which is how a block-compressed level 0 of three or four
        // channels is stored: BC5 carries two channels, so channels 0-1 are the first file and what is left is the
        // second. Each entry of "files" is an OBJECT carrying that file's own dxgi_format_id and channels_stored,
        // because the two need not agree; the values are checked against the .dds headers rather than trusted. An
        // entry that is a BARE STRING is the older spelling of the same key and is read as a name with nothing to
        // cross-check, which is what every other reader in this tree has always done.
        const JVal* files = t.get("files");
        if (files && files->kind == JVal::ARR) {
            if (files->arr.size() != 2) { fprintf(stderr, "ERROR: level %d: \"files\" must name two .dds files\n", l); return false; }
            std::string names[2];
            int want_dxgi[2] = { 0, 0 }, want_stored[2] = { 0, 0 };
            for (int i = 0; i < 2; i++) {
                const JVal& e = files->arr[(size_t)i];
                if (e.kind != JVal::OBJ && e.kind != JVal::STR) { fprintf(stderr, "ERROR: level %d: \"files\"[%d] must be a file name or an object carrying one\n", l, i); return false; }
                names[i] = e.kind == JVal::OBJ ? e.string("file") : e.str;
                if (names[i].empty()) { fprintf(stderr, "ERROR: level %d: \"files\"[%d] names no file\n", l, i); return false; }
                want_dxgi[i] = e.kind == JVal::OBJ ? (int)e.number("dxgi_format_id", 0) : 0;   // 0 = the older spelling: nothing published to cross-check
                want_stored[i] = e.kind == JVal::OBJ ? (int)e.number("channels_stored", 0) : 0;
            }
            if (!load_dds(beside(names[0]), L) || !load_dds(beside(names[1]), L, true)) return false;
            L.files = 2;
            const int got_dxgi[2] = { L.dxgi, L.dxgi_b }, got_stored[2] = { L.stored_C, L.stored_C_b };
            for (int i = 0; i < 2; i++)
                if ((want_dxgi[i] && want_dxgi[i] != got_dxgi[i]) || (want_stored[i] && want_stored[i] != got_stored[i])) {
                    fprintf(stderr, "ERROR: level %d: the JSON says %s is DXGI %d storing %d channel(s) and the file is DXGI %d storing %d\n",
                            l, names[i].c_str(), want_dxgi[i], want_stored[i], got_dxgi[i], got_stored[i]);
                    return false;
                }
        } else if (!load_dds(beside(t.string("file")), L)) return false;
        // The channels the level's files hold between them: one file stores stored_C, two store stored_C + stored_C_b
        // (2 + 1 for a three-channel BC level 0, which is exactly the three the decoder reads).
        const int stored_all = L.stored_C + (L.files > 1 ? L.stored_C_b : 0);
        if (L.C < 1 || L.C > 4 || L.C > stored_all) { fprintf(stderr, "ERROR: level %d: %d channels used of %d stored\n", l, L.C, stored_all); return false; }
        // The load-time pack, for an uncompressed level 0 only; key 4 binds it. It is not fatal when it fails - the
        // file's own plane is still there and is what the viewer shows - but it says so rather than leaving key 4 dead
        // with no explanation, and the raw levels it needed are dropped once it has run.
        if (l == 0 && !L.file_bc) {
            // A device with no textureCompressionBC cannot hold the pack at all, and that is not a fault in this run:
            // the asset is uncompressed, it draws, and the only thing missing is the comparison key 4 offers. So it is
            // one note rather than the three ERROR lines the attempt would otherwise leave behind in a run that
            // succeeds - the format refusal inside upload_image, the pack's own, and the warning here.
            if (!g_has_bc)
                printf("  note: this device does not report textureCompressionBC, so level 0 is not packed at load; "
                       "key 4 has nothing to bind\n");
            else if (!bc_pack_level0(L)) fprintf(stderr, "WARNING: level 0 was not packed at load; key 4 has nothing to bind\n");
            L.raw.clear(); L.raw.shrink_to_fit();
        }
        // Level 1's LOD shift and the one sampler table. The value is the JSON's own lod_bias_level1, which is what the
        // format says the level-1 sample must carry; log2(block) is only the value the writer puts there, and a file
        // that ever said something else would mean it.
        if (l == 1) {
            // Absent means log2(block), which is what every writer puts there. Present and not a number is refused, as
            // an out-of-range number is: it used to fall back to the default in silence.
            const JVal* lbv = root.get("lod_bias_level1");
            if (lbv && lbv->kind != JVal::NUM) {
                fprintf(stderr, "ERROR: the descriptor's lod_bias_level1 is not a number\n");
                return false;
            }
            const float lb = lbv ? (float)lbv->num : std::log2((float)(g.block > 1 ? g.block : 1));
            if (!check_level1_lod_shift(lb)) return false;
            g.lod_bias_level1 = lb;
            create_samplers();
        }
        const JVal* dq = t.get("dequantise"); const JVal* lo = dq ? dq->get("lo") : nullptr; const JVal* hi = dq ? dq->get("hi") : nullptr;
        if (!lo || !hi || lo->kind != JVal::ARR || hi->kind != JVal::ARR || (int)lo->arr.size() < L.C || (int)hi->arr.size() < L.C) { fprintf(stderr, "ERROR: level %d: no lo / hi per channel\n", l); return false; }
        for (int c = 0; c < L.C; c++) { L.lo[c] = (float)lo->arr[(size_t)c].num; L.hi[c] = (float)hi->arr[(size_t)c].num; }
        printf("  level %d: %d channel%s used, bits", l, L.C, L.C == 1 ? "" : "s"); for (int c = 0; c < L.C; c++) printf(" %d", L.bits[c]);
        printf(", dequantise value = lo + sample * (hi - lo) with lo / hi"); for (int c = 0; c < L.C; c++) printf(" [%g, %g]", (double)L.lo[c], (double)L.hi[c]); printf("\n");
    }
    const JVal* dec = root.get("decoder");
    if (!dec) { fprintf(stderr, "ERROR: no decoder in the JSON\n"); return false; }
    if (dec->string("type") != "bilinear") { fprintf(stderr, "ERROR: decoder type '%s': this viewer decodes the bilinear decoder only\n", dec->string("type").c_str()); return false; }
    const JVal* hidden = dec->get("hidden"); if (hidden && hidden->kind == JVal::ARR && !hidden->arr.empty()) { fprintf(stderr, "ERROR: hidden layers are not supported\n"); return false; }
    if (dec->string("output") != "identity") { fprintf(stderr, "ERROR: output rule '%s' is not the identity\n", dec->string("output").c_str()); return false; }
    const int C0 = (int)dec->number("C0"), C1 = (int)dec->number("C1"), nin = (int)dec->number("nin"), nout = (int)dec->number("nout");
    if (C0 != g.lat[0].C || C1 != g.lat[1].C || nin < 1 || nin > 24 || nout < 3 || nout > 18 || nout != 3 * g.textures_out) { fprintf(stderr, "ERROR: decoder shape C0 %d C1 %d nin %d nout %d does not fit the textures (%d / %d channels, %d outputs) or the shader's limits (nin <= 24, nout <= 18)\n", C0, C1, nin, nout, g.lat[0].C, g.lat[1].C, 3 * g.textures_out); return false; }
    g.terms = dec->string("terms"); int mask = 0;
    // The terms are a SEQUENCE, not a set: the shader builds phi in the order a, b, sc, so "b a sc" would name the same
    // three groups and describe a different feature vector. Anything out of that order is refused rather than reordered.
    { std::istringstream ss(g.terms); std::string w; int last = -1; while (ss >> w) { const int which = w == "a" ? 0 : (w == "b" ? 1 : (w == "sc" ? 2 : -1));
        if (which < 0) { fprintf(stderr, "ERROR: the term '%s' is not one the shader builds (a, b, sc)\n", w.c_str()); return false; }
        if (which <= last) { fprintf(stderr, "ERROR: the terms '%s' are not in the order the shader builds phi (a, then b, then sc)\n", g.terms.c_str()); return false; }
        last = which; mask |= 1 << which; } }
    int expect = 0; if (mask & 1) expect += C1; if (mask & 2) expect += C0; if (mask & 4) expect += C0 * C1;
    if (expect != nin) { fprintf(stderr, "ERROR: the terms '%s' give %d features, the decoder has nin %d\n", g.terms.c_str(), expect, nin); return false; }
    const JVal* layers = dec->get("layers");
    if (!layers || layers->kind != JVal::ARR || layers->arr.size() != 1) { fprintf(stderr, "ERROR: expected one layer\n"); return false; }
    const JVal& lay = layers->arr[0]; const JVal* Wv = lay.get("weights"); const JVal* bv = lay.get("bias");
    if ((int)lay.number("rows") != nout || (int)lay.number("cols") != nin || !Wv || !bv || (int)Wv->arr.size() != nout * nin || (int)bv->arr.size() != nout) { fprintf(stderr, "ERROR: the layer is not %d x %d\n", nout, nin); return false; }
    DecoderConstants& d = g.cb.dec; memset(&d, 0, sizeof(d));
    for (int c = 0; c < 4; c++) { d.lo0[c] = g.lat[0].lo[c]; d.hi0[c] = g.lat[0].hi[c]; d.lo1[c] = g.lat[1].lo[c]; d.hi1[c] = g.lat[1].hi[c]; }
    d.dims[0] = C0; d.dims[1] = C1; d.dims[2] = nin; d.dims[3] = nout; d.sel[0] = mask; d.sel[1] = 0;
    // The textures level 0 is read from: two when the file is two of them, and the shader's sel.z == 2 is what makes it
    // take channels 2-3 from the second. set_uniforms rewrites it every frame, because key 4 changes it.
    d.sel[2] = g.lat[0].files > 1 ? 2 : 0;
    for (int r = 0; r < nout; r++) for (int c = 0; c < nin; c++) d.W[r * 6 + c / 4][c % 4] = (float)Wv->arr[(size_t)r * (size_t)nin + (size_t)c].num;
    for (int r = 0; r < nout; r++) d.bias[r / 4][r % 4] = (float)bv->arr[(size_t)r].num;
    printf("  decoder: bilinear, terms '%s', phi %d -> %d outputs (%d texture%s), %d weights\n", g.terms.c_str(), nin, nout, g.textures_out, g.textures_out == 1 ? "" : "s", nout * nin + nout);
    return true;
}

// ---------------------------------------------------------------------------
// The descriptor set: one uniform buffer, three sampled images and two samplers, in set 0.
//
// The images and the samplers are kept SEPARATE rather than combined, which is what makes the key handling of stage 3
// trivial: a key that changes the filter, the mips or the bias rewrites one or two sampler descriptors and touches
// nothing else. In the shader that is texture(sampler2D(lat0, samp), uv), which is ordinary Vulkan GLSL.
// ---------------------------------------------------------------------------
static VkDescriptorSetLayout g_set_layout = VK_NULL_HANDLE;
static VkDescriptorPool      g_descriptor_pool = VK_NULL_HANDLE;
static VkDescriptorSet       g_descriptor_set = VK_NULL_HANDLE;
static VkPipelineLayout      g_pipeline_layout = VK_NULL_HANDLE;
static VkPipeline            g_pipeline = VK_NULL_HANDLE;
static VkPipeline            g_pipeline_cv = VK_NULL_HANDLE;   // the cooperative-vector scene pipeline, or null
static VkFormat              g_pipeline_format = VK_FORMAT_UNDEFINED;
static VkBuffer              g_uniform_buffer = VK_NULL_HANDLE;
static VkDeviceMemory        g_uniform_memory = VK_NULL_HANDLE;
static void*                 g_uniform_mapped = nullptr;

static void create_descriptors(void) {
    VkDescriptorSetLayoutBinding b[7]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; b[0].descriptorCount = 1;
    b[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    for (uint32_t i = 1; i <= 3; i++) {
        b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    for (uint32_t i = 4; i <= 5; i++) {
        b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    // Binding 6, the converted matrix and the bias, exists ONLY on a device the stage-4 query passed on. A layout that
    // declared it everywhere would be a change to the baseline path's descriptor set for the sake of a pipeline that
    // cannot be built, and the baseline path is the product.
    const uint32_t bindings = g_cv_available ? 7u : 6u;
    if (g_cv_available) {
        b[6].binding = 6; b[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b[6].descriptorCount = 1;
        b[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = bindings; li.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(g_device, &li, nullptr, &g_set_layout));

    const VkDescriptorPoolSize sizes[4] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3 },
        { VK_DESCRIPTOR_TYPE_SAMPLER, 2 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
    };
    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.maxSets = 1; pi.poolSizeCount = g_cv_available ? 4u : 3u; pi.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(g_device, &pi, nullptr, &g_descriptor_pool));

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = g_descriptor_pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &g_set_layout;
    VK_CHECK(vkAllocateDescriptorSets(g_device, &ai, &g_descriptor_set));

    // The uniform buffer is persistently mapped and host coherent, and the frame waits on the device before it writes:
    // one frame in flight is what makes that safe with no double buffering of anything.
    create_buffer(sizeof(Uniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  g_uniform_buffer, g_uniform_memory);
    VK_CHECK(vkMapMemory(g_device, g_uniform_memory, 0, sizeof(Uniforms), 0, &g_uniform_mapped));

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &g_set_layout;
    VK_CHECK(vkCreatePipelineLayout(g_device, &pli, nullptr, &g_pipeline_layout));
}

// What the frame reads: the three images and the two samplers the current state picks. It is a function rather than a
// line at load because stage 3's keys rewrite exactly this.
static void update_descriptors(void) {
    // Key 4: the load-time pack's textures in place of the file's own, when there is a pack. The third slot is level
    // 0's channels past the first two wherever they live in a texture of their own - the file's second one, or the
    // pack's - and the 1x1 dummy when they do not, because Vulkan has no null descriptor here.
    const bool packed = g_bc && g.lat[0].bc_n > 0;
    const VkImageView views[3] = {
        packed ? g.lat[0].bc_img[0].view : g.lat[0].img.view,
        g.lat[1].img.view,
        packed ? (g.lat[0].bc_n > 1 ? g.lat[0].bc_img[1].view : g_dummy.view)
               : (g.lat[0].files > 1 ? g.lat[0].img_b.view : g_dummy.view),
    };
    const int slot = filter_slot();
    const int mips = g.mips_on ? 1 : 0;
    // s0 level 0, s1 level 1: the SAME state. Level 1's LOD shift is const0.z, the gradient scale, not a sampler bias.
    const VkSampler samplers[2] = { g_samplers[mips][slot], g_samplers[mips][slot] };

    VkDescriptorBufferInfo bi{};
    bi.buffer = g_uniform_buffer; bi.offset = 0; bi.range = sizeof(Uniforms);
    VkDescriptorImageInfo ii[3]{};
    for (int i = 0; i < 3; i++) { ii[i].imageView = views[i]; ii[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; }
    VkDescriptorImageInfo si[2]{};
    for (int i = 0; i < 2; i++) si[i].sampler = samplers[i];

    VkWriteDescriptorSet w[6]{};
    for (uint32_t i = 0; i < 6; i++) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = g_descriptor_set; w[i].dstBinding = i; w[i].descriptorCount = 1;
    }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &bi;
    for (uint32_t i = 1; i <= 3; i++) { w[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[i].pImageInfo = &ii[i - 1]; }
    for (uint32_t i = 4; i <= 5; i++) { w[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[i].pImageInfo = &si[i - 4]; }
    vkUpdateDescriptorSets(g_device, 6, w, 0, nullptr);
}

// ---------------------------------------------------------------------------
// The converted matrix and the bias, built once at load and never touched again.
//
// The matrix CANNOT go into the buffer as the descriptor's row-major fp32 and be read in the inferencing-optimal
// layout: the optimal layouts are implementation-defined and the only thing that may produce one is the extension's
// own conversion. vkConvertCooperativeVectorMatrixNV is called twice, once with a null destination to learn the size
// the driver wants and once to fill it, and the one call does the layout change and the fp32 -> fp16 conversion
// together. The HOST variant is used rather than the command-buffer one because this happens at load, on data that is
// already in host memory, and nothing is waiting on it.
//
// The offsets are the extension's rule and not a preference: the matrix must start 64-byte aligned and the bias
// 16-byte aligned, and those requirements apply to the base of the buffer too - so the matrix sits at offset 0 of a
// buffer of its own and the bias at the next 64-byte boundary after it.
// ---------------------------------------------------------------------------
#if NNTC_HAVE_COOPVEC
static bool build_cooperative_vector_weights(void) {
    if (!g_cv_available) return true;

    // The padded 18 by 24, row-major, from the same DecoderConstants the plain shader reads: the rows past nout and
    // the columns past nin are the zeros memset left there, which is what makes the padded product exact.
    std::vector<float> src((size_t)CV_M * CV_K, 0.0f);
    const DecoderConstants& d = g.cb.dec;
    for (uint32_t r = 0; r < CV_M; r++)
        for (uint32_t c = 0; c < CV_K; c++) src[(size_t)r * CV_K + c] = d.W[r * 6 + c / 4][c % 4];
    std::vector<float> bias(CV_M, 0.0f);
    for (uint32_t r = 0; r < CV_M; r++) bias[r] = d.bias[r / 4][r % 4];

    // THE RANGE GUARD. Every tuple this viewer can use reads the matrix as fp16, whose largest finite value is 65504,
    // and this asset's weights are read off a file rather than produced here. A weight past that becomes an infinity
    // in the conversion and the decode becomes NaN over the whole quad - a picture that is wrong rather than slightly
    // wrong - so an asset whose decoder does not fit the type is refused the second pipeline and drawn by the plain
    // one, which reads W as the fp32 it is. (The small end needs no guard: fp16 subnormals go down to 6e-8 and a
    // weight below even that becomes a zero, which is a rounding error of the size the frames are compared at.)
    float largest = 0.0f;
    for (float v : src) if (fabsf(v) > largest) largest = fabsf(v);
    for (float v : bias) if (fabsf(v) > largest) largest = fabsf(v);
    if (largest > 65504.0f) {
        char why[192];
        snprintf(why, sizeof(why), "this asset's decoder has a weight of %g, past fp16's largest finite value of "
                                   "65504, and every usable type tuple reads the matrix as fp16", (double)largest);
        g_cv_why_not = why;
        g_cv_available = false;
        printf("cooperative vectors: not available - %s\n", g_cv_why_not.c_str());
        if (g_cv_want == 1) {
            fprintf(stderr, "ERROR: --coopvec 1 was asked for and this asset cannot: %s\n", g_cv_why_not.c_str());
            return false;
        }
        g_cv_on = false;
        // The descriptor set was built with binding 6 in it, before the weights were read and before this could be
        // known, so the binding is written with a small zeroed buffer rather than left unwritten: no shader that is
        // built now reads it, and a set whose every binding is written is one less thing for the layer to have an
        // opinion about.
        g_cv_bytes = 64;
        create_buffer(g_cv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      g_cv_buffer, g_cv_memory);
        VK_CHECK(vkMapMemory(g_device, g_cv_memory, 0, g_cv_bytes, 0, &g_cv_mapped));
        memset(g_cv_mapped, 0, (size_t)g_cv_bytes);
        VkDescriptorBufferInfo zbi{};
        zbi.buffer = g_cv_buffer; zbi.offset = 0; zbi.range = g_cv_bytes;
        VkWriteDescriptorSet zw{};
        zw.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        zw.dstSet = g_descriptor_set; zw.dstBinding = 6; zw.descriptorCount = 1;
        zw.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; zw.pBufferInfo = &zbi;
        vkUpdateDescriptorSets(g_device, 1, &zw, 0, nullptr);
        printf("decode: plain (%s)\n", g_cv_why_not.c_str());
        return true;
    }

    // The bias is read according to the tuple's biasInterpretation and nothing converts it on the way in, so it
    // goes into the buffer in that type: fp32 under the preferred tuple, fp16 under the all-fp16 one.
    std::vector<uint16_t> bias16(CV_M, 0);
    if (!g_cv_fp32) for (uint32_t r = 0; r < CV_M; r++) bias16[r] = float_to_half(bias[r]);
    const void* bias_bytes = g_cv_fp32 ? (const void*)bias.data() : (const void*)bias16.data();
    const size_t bias_size = g_cv_fp32 ? bias.size() * sizeof(float) : bias16.size() * sizeof(uint16_t);

    VkConvertCooperativeVectorMatrixInfoNV info{};
    info.sType = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV;
    info.srcSize = src.size() * sizeof(float);
    info.srcData.hostAddress = src.data();
    info.srcComponentType = VK_COMPONENT_TYPE_FLOAT32_KHR;
    info.dstComponentType = VK_COMPONENT_TYPE_FLOAT16_KHR;
    info.numRows = CV_M; info.numColumns = CV_K;
    info.srcLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV;
    // The stride is exactly one row and not one row plus padding, which reads oddly against
    // VUID-VkConvertCooperativeVectorMatrixInfoNV-srcLayout-10077: for a row-major source it requires srcStride to be
    // "greater than" the row's length in bytes, which taken literally would forbid a tightly packed matrix. The
    // validation layer accepts stride == row length, which is what the requirement is plainly meant to say (a stride
    // shorter than a row would make rows overlap), and a tightly packed 18 by 24 is what the asset's W already is.
    info.srcStride = (size_t)CV_K * sizeof(float);
    info.dstLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV;
    info.dstStride = 0;   // ignored for an optimal layout, and the driver decides the whole of it

    size_t matrix_bytes = 0;
    info.pDstSize = &matrix_bytes;
    info.dstData.hostAddress = nullptr;
    VK_CHECK(g_cv_convert(g_device, &info));
    if (!matrix_bytes) {
        fprintf(stderr, "ERROR: the driver asked for a zero-byte cooperative-vector matrix\n");
        return false;
    }
    std::vector<uint8_t> converted(matrix_bytes);
    info.dstData.hostAddress = converted.data();
    VK_CHECK(g_cv_convert(g_device, &info));

    g_cv_bias_offset = (matrix_bytes + 63) & ~(VkDeviceSize)63;
    // The buffer is rounded UP to a multiple of 64 bytes and the tail is left zero. The bias is eighteen entries and
    // the shader fetches it as a cooperative vector of 18, which an implementation is free to read in 16-byte units:
    // a buffer that ended exactly at the bias's own length would let the last of those units run past the end of the
    // allocation. Rounding costs at most 63 bytes once.
    g_cv_bytes = (g_cv_bias_offset + bias_size + 63) & ~(VkDeviceSize)63;
    create_buffer(g_cv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  g_cv_buffer, g_cv_memory);
    VK_CHECK(vkMapMemory(g_device, g_cv_memory, 0, g_cv_bytes, 0, &g_cv_mapped));
    memset(g_cv_mapped, 0, (size_t)g_cv_bytes);
    memcpy(g_cv_mapped, converted.data(), converted.size());
    memcpy((uint8_t*)g_cv_mapped + g_cv_bias_offset, bias_bytes, bias_size);

    // Written once: nothing about the weights changes while the program runs, so unlike the images and the samplers
    // this descriptor is not part of what update_descriptors rewrites every frame.
    VkDescriptorBufferInfo bi{};
    bi.buffer = g_cv_buffer; bi.offset = 0; bi.range = g_cv_bytes;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = g_descriptor_set; w.dstBinding = 6; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &bi;
    vkUpdateDescriptorSets(g_device, 1, &w, 0, nullptr);

    printf("  cooperative vectors: the %ux%u matrix converted to the device's inferencing-optimal fp16 layout, "
           "%u bytes, with the %s bias at offset %u\n", (unsigned)CV_M, (unsigned)CV_K, (unsigned)matrix_bytes,
           g_cv_fp32 ? "fp32" : "fp16", (unsigned)g_cv_bias_offset);
    return true;
}
#endif

// ---------------------------------------------------------------------------
// The shaders: GLSL compiled AT RUNTIME by shaderc, which the Vulkan SDK ships as a static library. That is the direct
// analogue of the Direct3D viewer's D3DCompileFromFile, and it is what lets the shader be edited beside the executable
// and reloaded - the key for that is stage 3, but the mechanism is here and the files travel with the build.
// ---------------------------------------------------------------------------
// `define` is one preprocessor macro or nullptr. It exists for the cooperative-vector shader, which is ONE file
// compiled for whichever type tuple the device enumerated rather than two files that would drift apart.
static bool compile_glsl_source(const std::string& source, const std::string& path, shaderc_shader_kind kind,
                                std::vector<uint32_t>& spv, const char* define = nullptr,
                                const char* value = nullptr) {
    shaderc_compiler_t compiler = shaderc_compiler_initialize();
    if (!compiler) { fprintf(stderr, "ERROR: the shader compiler could not be created\n"); return false; }
    shaderc_compile_options_t options = shaderc_compile_options_initialize();
    if (!options) { shaderc_compiler_release(compiler); fprintf(stderr, "ERROR: the shader compiler options could not be created\n"); return false; }
    shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
    if (define) shaderc_compile_options_add_macro_definition(options, define, strlen(define), value,
                                                            value ? strlen(value) : 0);
    shaderc_compilation_result_t result = shaderc_compile_into_spv(compiler, source.data(), source.size(), kind,
                                                                   path.c_str(), "main", options);
    const bool ok = result && shaderc_result_get_compilation_status(result) == shaderc_compilation_status_success;
    if (result) {
        const char* message = shaderc_result_get_error_message(result);
        if (message && *message) fprintf(stderr, "%s:\n%s\n", ok ? "SHADER WARNING" : "SHADER ERROR", message);
        if (ok) {
            const size_t bytes = shaderc_result_get_length(result);
            spv.assign((const uint32_t*)shaderc_result_get_bytes(result),
                       (const uint32_t*)(shaderc_result_get_bytes(result) + bytes));
        }
        shaderc_result_release(result);
    } else {
        fprintf(stderr, "ERROR: the shader '%s' could not be compiled\n", path.c_str());
    }
    shaderc_compile_options_release(options);
    shaderc_compiler_release(compiler);
    return ok;
}

// The same, for a shader that lives in a file: the scene's two, which the build copies beside the executable and key R
// re-reads from disk. The overlay's pair are string constants and go through the function above, exactly as the other
// viewer's DEBUG_HLSL does.
static bool compile_glsl(const std::string& path, shaderc_shader_kind kind, std::vector<uint32_t>& spv,
                         const char* define = nullptr, const char* value = nullptr) {
    const std::string source = read_file(path);
    if (source.empty()) { fprintf(stderr, "ERROR: cannot read the shader '%s'\n", path.c_str()); return false; }
    return compile_glsl_source(source, path, kind, spv, define, value);
}

static VkShaderModule make_module(const std::vector<uint32_t>& spv) {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spv.size() * sizeof(uint32_t); ci.pCode = spv.data();
    VkShaderModule m = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(g_device, &ci, nullptr, &m));
    return m;
}

// The directory the running executable is in. It is asked of the OPERATING SYSTEM on both platforms rather than
// derived from argv[0], which is only ever a path when the caller typed one: a program started through the PATH is
// handed its bare name, and deriving a directory from that yields nothing - which would send the shader search to the
// working directory, the one place this viewer must never read a shader from (see shader_path below). Windows asks
// the module; Linux reads /proc/self/exe, which the kernel keeps as the resolved path of the running image.
//
// argv[0] is the fallback and only that, for a system with no /proc mounted. When neither knows, the viewer refuses:
// it cannot find its shaders, and guessing at the working directory is exactly the silent wrong answer this rule
// exists to prevent.
static std::string exe_dir(const char* argv0) {
#ifdef _WIN32
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, (DWORD)sizeof(buf));
    if (n > 0 && n < (DWORD)sizeof(buf)) {
        const std::string exe(buf, buf + n);
        const size_t s = exe.find_last_of("/\\");
        if (s != std::string::npos) return exe.substr(0, s + 1);
    }
#else
    {
        char buf[4096];
        const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (n > 0 && (size_t)n < sizeof(buf)) {
            const std::string exe(buf, buf + n);
            const size_t s = exe.find_last_of('/');
            if (s != std::string::npos) return exe.substr(0, s + 1);
        }
    }
#endif
    const std::string exe = argv0 ? argv0 : "";
    const size_t s = exe.find_last_of("/\\");
    if (s == std::string::npos) {
        fprintf(stderr, "ERROR: this program cannot tell which directory it is running from, so it cannot find "
                        "view.vert and view.frag beside itself; start it by its path rather than by name\n");
        exit(1);
    }
    return exe.substr(0, s + 1);
}

// The shader BESIDE THE EXECUTABLE, and nowhere else. It used to try the working directory first, which meant that a
// stray view.frag anywhere the viewer happened to be started from silently replaced the real one - a blank frame and
// an exit code of 0, which is the worst shape a failure can take. The build copies the three shader files next to the
// executable on every build, so that is the one place they are read from, by --shot and by key R alike; the full path is printed
// on every compile so there is never any doubt which file was read. That rule also closes the hole a shader with no
// entry point would open, and shaderc closes the rest of it: glslang reports a stage with no main as a compilation
// error ("Missing entry point"), so a file that compiles at all has one.
static std::string shader_path(const char* argv0, const char* name) {
    return exe_dir(argv0) + name;
}

static std::string g_vert_path, g_frag_path, g_frag_cv_path;

// ---------------------------------------------------------------------------
// The overlay's own objects. It is a second pipeline over four vertices with its own one-binding descriptor set, and
// its two shaders are STRING CONSTANTS rather than files: they are two lines each and there is nothing in them to
// edit, exactly as the other viewer's DEBUG_HLSL is a constant while its scene shader is a file key R re-reads.
// ---------------------------------------------------------------------------
static GpuImage       g_overlay_image;
static VkBuffer       g_overlay_staging = VK_NULL_HANDLE;
static VkDeviceMemory g_overlay_staging_memory = VK_NULL_HANDLE;
static void*          g_overlay_mapped = nullptr;
static bool           g_overlay_upload = false;     // the strip's bytes have changed and the next frame must copy them
static bool           g_overlay_uploaded = false;   // the image has been written once, so its old layout is not UNDEFINED any more
static VkSampler      g_overlay_sampler = VK_NULL_HANDLE;
static VkDescriptorSetLayout g_overlay_set_layout = VK_NULL_HANDLE;
static VkDescriptorPool      g_overlay_pool = VK_NULL_HANDLE;
static VkDescriptorSet       g_overlay_set = VK_NULL_HANDLE;
static VkPipelineLayout      g_overlay_pipeline_layout = VK_NULL_HANDLE;
static VkPipeline            g_overlay_pipeline = VK_NULL_HANDLE;
static VkBuffer       g_overlay_geometry = VK_NULL_HANDLE;
static VkDeviceMemory g_overlay_geometry_memory = VK_NULL_HANDLE;
static void*          g_overlay_geometry_mapped = nullptr;
static const VkDeviceSize OVERLAY_INDEX_OFFSET = 4 * 4 * sizeof(float);   // four vertices of a position and a uv

static const char* const OVERLAY_VERT =
    "#version 450\n"
    "layout(location = 0) in vec2 in_pos;\n"
    "layout(location = 1) in vec2 in_uv;\n"
    "layout(location = 0) out vec2 v_uv;\n"
    "void main() { gl_Position = vec4(in_pos, 0.0, 1.0); v_uv = in_uv; }\n";
static const char* const OVERLAY_FRAG =
    "#version 450\n"
    "layout(set = 0, binding = 0) uniform sampler2D strip;\n"
    "layout(location = 0) in vec2 v_uv;\n"
    "layout(location = 0) out vec4 out_colour;\n"
    "void main() { out_colour = texture(strip, v_uv); }\n";

// The image, its staging buffer, its sampler, its descriptor set and its four vertices. Everything here is created
// once and rewritten in place, which one frame in flight makes safe.
static void init_overlay(void) {
    VkImageCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D; ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent = { (uint32_t)OVL_W, (uint32_t)OVL_TEX_H, 1 }; ii.mipLevels = 1; ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE; ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(g_device, &ii, nullptr, &g_overlay_image.image));
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g_device, g_overlay_image.image, &req);
    VkMemoryAllocateInfo ma{};
    ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ma.allocationSize = req.size;
    ma.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(g_device, &ma, nullptr, &g_overlay_image.memory));
    VK_CHECK(vkBindImageMemory(g_device, g_overlay_image.image, g_overlay_image.memory, 0));
    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = g_overlay_image.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = VK_FORMAT_R8G8B8A8_UNORM;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VK_CHECK(vkCreateImageView(g_device, &vi, nullptr, &g_overlay_image.view));

    create_buffer((VkDeviceSize)OVL_W * OVL_TEX_H * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  g_overlay_staging, g_overlay_staging_memory);
    VK_CHECK(vkMapMemory(g_device, g_overlay_staging_memory, 0, (VkDeviceSize)OVL_W * OVL_TEX_H * 4, 0, &g_overlay_mapped));

    // A point sampler with no mips: the strip is drawn at its own size and a filtered fetch would only blur the glyphs.
    // It is created here rather than taken from the scene's eight, because those carry the keys' state.
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.compareOp = VK_COMPARE_OP_NEVER;
    si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    VK_CHECK(vkCreateSampler(g_device, &si, nullptr, &g_overlay_sampler));

    VkDescriptorSetLayoutBinding b{};
    b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 1; li.pBindings = &b;
    VK_CHECK(vkCreateDescriptorSetLayout(g_device, &li, nullptr, &g_overlay_set_layout));
    const VkDescriptorPoolSize size = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.maxSets = 1; pi.poolSizeCount = 1; pi.pPoolSizes = &size;
    VK_CHECK(vkCreateDescriptorPool(g_device, &pi, nullptr, &g_overlay_pool));
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = g_overlay_pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &g_overlay_set_layout;
    VK_CHECK(vkAllocateDescriptorSets(g_device, &ai, &g_overlay_set));
    VkDescriptorImageInfo di{};
    di.sampler = g_overlay_sampler; di.imageView = g_overlay_image.view;
    di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = g_overlay_set; w.dstBinding = 0; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &di;
    vkUpdateDescriptorSets(g_device, 1, &w, 0, nullptr);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &g_overlay_set_layout;
    VK_CHECK(vkCreatePipelineLayout(g_device, &pli, nullptr, &g_overlay_pipeline_layout));

    create_buffer(OVERLAY_INDEX_OFFSET + 6 * sizeof(uint32_t),
                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  g_overlay_geometry, g_overlay_geometry_memory);
    VK_CHECK(vkMapMemory(g_device, g_overlay_geometry_memory, 0, OVERLAY_INDEX_OFFSET + 6 * sizeof(uint32_t), 0,
                         &g_overlay_geometry_mapped));
    const uint32_t indices[6] = { 0, 1, 2, 0, 2, 3 };
    memcpy((uint8_t*)g_overlay_geometry_mapped + OVERLAY_INDEX_OFFSET, indices, sizeof(indices));
}

// The overlay's pipeline: the same dynamic-rendering setup as the scene's, with alpha blending on, the depth test off
// and a two-vec2 vertex. It is rebuilt with the scene's, because both name the colour attachment's format.
//
// It BUILDS and hands back, and destroys nothing: what is live is only replaced once both pipelines have been made, so
// that a failed key R leaves the picture exactly as it was.
static bool build_overlay_pipeline(VkFormat colour_format, VkPipeline& out) {
    std::vector<uint32_t> vs_spv, fs_spv;
    if (!compile_glsl_source(OVERLAY_VERT, "overlay.vert", shaderc_vertex_shader, vs_spv)) return false;
    if (!compile_glsl_source(OVERLAY_FRAG, "overlay.frag", shaderc_fragment_shader, fs_spv)) return false;
    VkShaderModule vs = make_module(vs_spv), fs = make_module(fs_spv);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vs; stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fs; stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0; binding.stride = 4 * sizeof(float); binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription attrs[2]{};
    attrs[0].location = 0; attrs[0].binding = 0; attrs[0].format = VK_FORMAT_R32G32_SFLOAT; attrs[0].offset = 0;
    attrs[1].location = 1; attrs[1].binding = 0; attrs[1].format = VK_FORMAT_R32G32_SFLOAT; attrs[1].offset = 2 * sizeof(float);
    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = 2; vi.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_FALSE; ds.depthWriteEnable = VK_FALSE; ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.colorBlendOp = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba.alphaBlendOp = VK_BLEND_OP_ADD;
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1; cb.pAttachments = &cba;
    const VkDynamicState dynamic[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy{};
    dy.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dy.dynamicStateCount = 2; dy.pDynamicStates = dynamic;
    VkPipelineRenderingCreateInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    ri.colorAttachmentCount = 1; ri.pColorAttachmentFormats = &colour_format;
    ri.depthAttachmentFormat = g_depth_format;

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.pNext = &ri;
    pi.stageCount = 2; pi.pStages = stages;
    pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
    pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb; pi.pDynamicState = &dy;
    pi.layout = g_overlay_pipeline_layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult made = vkCreateGraphicsPipelines(g_device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline);
    vkDestroyShaderModule(g_device, vs, nullptr);
    vkDestroyShaderModule(g_device, fs, nullptr);
    if (made != VK_SUCCESS) {
        fprintf(stderr, "ERROR: the overlay pipeline could not be created (%s)\n", vk_result_name(made));
        return false;
    }
    out = pipeline;
    return true;
}

// The scene's pipeline, built from the two shader files. Like the overlay's it only builds: see create_pipeline below.
// One scene pipeline from one fragment shader. `frag_path` is view.frag for the plain path and view_coopvec.frag for
// the cooperative-vector one; everything else about the two pipelines is identical, which is what makes the two frames
// comparable. The second one carries a specialization constant, the byte offset of the bias in the weights buffer,
// because only the host knows how large the driver's own matrix layout turned out to be.
static bool build_scene_pipeline(const std::string& frag_path, bool coopvec, VkFormat colour_format, VkPipeline& out) {
    std::vector<uint32_t> vs_spv, fs_spv;
    if (!compile_glsl(g_vert_path, shaderc_vertex_shader, vs_spv)) return false;
    if (!compile_glsl(frag_path, shaderc_fragment_shader, fs_spv, coopvec ? "NNTC_CV_FP32" : nullptr,
                      g_cv_fp32 ? "1" : "0")) return false;
    VkShaderModule vs = make_module(vs_spv), fs = make_module(fs_spv);

    const uint32_t bias_offset = (uint32_t)g_cv_bias_offset;
    VkSpecializationMapEntry entry{};
    entry.constantID = 0; entry.offset = 0; entry.size = sizeof(uint32_t);
    VkSpecializationInfo spec{};
    spec.mapEntryCount = 1; spec.pMapEntries = &entry;
    spec.dataSize = sizeof(bias_offset); spec.pData = &bias_offset;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vs; stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fs; stages[1].pName = "main";
    if (coopvec) stages[1].pSpecializationInfo = &spec;

    // The same 20-byte vertex the other viewer uses: a float3 position and a float2 texture coordinate, interleaved.
    VkVertexInputBindingDescription binding{};
    binding.binding = 0; binding.stride = 5 * sizeof(float); binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription attrs[2]{};
    attrs[0].location = 0; attrs[0].binding = 0; attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[0].offset = 0;
    attrs[1].location = 1; attrs[1].binding = 0; attrs[1].format = VK_FORMAT_R32G32_SFLOAT; attrs[1].offset = 3 * sizeof(float);
    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = 2; vi.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.scissorCount = 1;   // both dynamic: a resize needs no pipeline rebuild

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;   // the other viewer's D3D11_CULL_NONE: the cube is readable from inside
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable = VK_FALSE;
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1; cb.pAttachments = &cba;

    const VkDynamicState dynamic[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy{};
    dy.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dy.dynamicStateCount = 2; dy.pDynamicStates = dynamic;

    // Dynamic rendering: the attachment formats go on the pipeline and there is no render pass to name.
    VkPipelineRenderingCreateInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    ri.colorAttachmentCount = 1; ri.pColorAttachmentFormats = &colour_format;
    ri.depthAttachmentFormat = g_depth_format;

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.pNext = &ri;
    pi.stageCount = 2; pi.pStages = stages;
    pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
    pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb; pi.pDynamicState = &dy;
    pi.layout = g_pipeline_layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult made = vkCreateGraphicsPipelines(g_device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline);
    vkDestroyShaderModule(g_device, vs, nullptr);
    vkDestroyShaderModule(g_device, fs, nullptr);
    if (made != VK_SUCCESS) {
        fprintf(stderr, "ERROR: the pipeline could not be created (%s)\n", vk_result_name(made));
        return false;
    }
    out = pipeline;
    return true;
}

// Both pipelines, and the only place either of them is replaced. BOTH are built into locals and BOTH are committed
// only when both succeeded: key R's promise is that a shader that will not compile leaves the picture where it was,
// and a version of this that committed the scene's pipeline before attempting the overlay's could leave the program
// running half of an edit if the second one failed. The old pair is destroyed after the new one is in place.
static bool create_pipeline(VkFormat colour_format) {
    VkPipeline scene = VK_NULL_HANDLE, coopvec = VK_NULL_HANDLE, overlay = VK_NULL_HANDLE;
    if (!build_scene_pipeline(g_frag_path, false, colour_format, scene)) return false;
    // The second scene pipeline, on a device the query passed on. It is part of the same all-or-nothing rule: a
    // cooperative-vector shader that will not compile leaves BOTH pictures where they were rather than leaving the
    // program with one of the two paths half replaced.
    if (g_cv_available && !build_scene_pipeline(g_frag_cv_path, true, colour_format, coopvec)) {
        vkDestroyPipeline(g_device, scene, nullptr);
        return false;
    }
    if (!build_overlay_pipeline(colour_format, overlay)) {
        vkDestroyPipeline(g_device, scene, nullptr);
        if (coopvec) vkDestroyPipeline(g_device, coopvec, nullptr);
        return false;
    }
    if (g_pipeline) vkDestroyPipeline(g_device, g_pipeline, nullptr);
    if (g_pipeline_cv) vkDestroyPipeline(g_device, g_pipeline_cv, nullptr);
    if (g_overlay_pipeline) vkDestroyPipeline(g_device, g_overlay_pipeline, nullptr);
    g_pipeline = scene;
    g_pipeline_cv = coopvec;
    g_overlay_pipeline = overlay;
    g_pipeline_format = colour_format;
    if (g_cv_available) printf("shaders compiled: %s, %s, %s\n", g_vert_path.c_str(), g_frag_path.c_str(),
                               g_frag_cv_path.c_str());
    else printf("shaders compiled: %s, %s\n", g_vert_path.c_str(), g_frag_path.c_str());
    return true;
}

// ---------------------------------------------------------------------------
// The overlay: the Direct3D viewer's four-line debug strip, the same font at the same scale, the same layout and the
// same words, rasterised on the cpu into an OVL_W x OVL_H RGBA buffer and drawn as one alpha-blended quad at the top
// left. Nothing about the rasteriser is api-specific, so blit_char and update_debug_text below ARE that viewer's;
// what differs is the upload (a staging buffer and one vkCmdCopyBufferToImage at the top of the frame) and the draw (a
// second pipeline with blending on and the depth test off).
// ---------------------------------------------------------------------------
static void blit_char(std::vector<uint8_t>& buf, int px, int py, char ch) {
    int c = (unsigned char)ch;
    if (c < 32 || c > 127) c = '.';
    const uint8_t* glyph = g_font8x8[c - 32];
    for (int y = 0; y < 8; y++) for (int x = 0; x < 8; x++) {
        if (!((glyph[y] >> x) & 1)) continue;
        for (int sy = 0; sy < FONT_SCALE; sy++) for (int sx = 0; sx < FONT_SCALE; sx++) {
            const int X = px + x * FONT_SCALE + sx, Y = py + y * FONT_SCALE + sy;
            if (X < 0 || X >= OVL_W || Y < 0 || Y >= OVL_TEX_H) continue;
            uint8_t* d = &buf[((size_t)Y * OVL_W + X) * 4];
            d[0] = 255; d[1] = 255; d[2] = 255; d[3] = 255;
        }
    }
}

// The stored format of a texture, by its DXGI id, as the strip names it.
static const char* format_name(int dxgi) {
    return dxgi == 61 ? "R8" : (dxgi == 49 ? "R8G8" : (dxgi == 28 ? "RGBA8" : (dxgi == 80 ? "BC4" : (dxgi == 83 ? "BC5" : "?"))));
}

static void update_debug_text(void) {
    if (!g.debug_dirty) return;
    std::vector<uint8_t> buf((size_t)OVL_W * OVL_TEX_H * 4);
    for (size_t i = 0; i < (size_t)OVL_W * OVL_H * 4; i += 4) { buf[i] = 0; buf[i + 1] = 0; buf[i + 2] = 0; buf[i + 3] = 180; }
    // The help page's backdrop is only as wide as its longest line; below the strip the rest stays transparent.
    int help_w = 0;
    for (int i = 0; i < HELP_N; i++) help_w = std::max(help_w, (int)strlen(HELP_LINES[i]));
    help_w = std::min(OVL_W, 8 + help_w * 8 * FONT_SCALE);
    for (int y = OVL_H; y < OVL_TEX_H; y++) for (int x = 0; x < help_w; x++) buf[((size_t)y * OVL_W + x) * 4 + 3] = 200;
    char l0[256], l1[256], l2[256];
    // The load-time pack's own name: one BC4 or BC5, and for three or four channels the second texture's format too
    // (three channels put channel 2 alone in a BC4, four put channels 2-3 in a second BC5).
    const char* packname = g.lat[0].C == 1 ? "BC4" : (g.lat[0].C == 2 ? "BC5" : (g.lat[0].C == 3 ? "BC5+BC4" : "BC5+BC5"));
    char bcs[64];
    if (g.lat[0].bc_n == 0) snprintf(bcs, sizeof(bcs), "U8");
    else if (g.lat[0].bc_lossless) snprintf(bcs, sizeof(bcs), "%s lossless", packname);
    else snprintf(bcs, sizeof(bcs), "%s %.1fdB", packname, g.lat[0].bc_psnr);
    // The format line when level 0 arrived block-compressed in the file: nothing was packed here, so it says "(file)"
    // and no quality.
    char bcf[64];
    snprintf(bcf, sizeof(bcf), "%s%s%s (file)", format_name(g.lat[0].dxgi), g.lat[0].files > 1 ? "+" : "",
             g.lat[0].files > 1 ? format_name(g.lat[0].dxgi_b) : "");
    // GPU memory of what is bound, in bytes, summed over the base and over the whole chain, as bits per source pixel of
    // the decode size per material texture. An uncompressed texture is one byte per stored channel per texel; a
    // BLOCK-COMPRESSED one is whole 4x4 blocks - ceil(w/4) * ceil(h/4) of them at 8 bytes (BC4) or 16 (BC5) - which is
    // what the driver allocates and what the .dds holds. Counting it as w * h * 0.5 would understate every level whose
    // size is not a multiple of 4, and the deep end of a chain is nearly all such levels.
    double mem_base = 0, mem_all = 0, mem_base_l0 = 0;
    for (int l = 0; l < 2; l++) {
        const Latent& L = g.lat[l];
        const bool packed = l == 0 && g_bc && L.bc_n > 0;
        // Each bound texture counted at ITS OWN rate, because the two textures of a three-channel level 0 are a BC5 and
        // a BC4 and not two of either: 16 + 8 bytes a block, which is the 12 bpp the file costs.
        double bytes_per_block = 0.0;   // zero when nothing here is block-compressed
        if (packed) { for (int p = 0; p < L.bc_n; p++) bytes_per_block += L.bc_nc[p] == 1 ? 8.0 : 16.0; }
        else if (L.file_bc) bytes_per_block = (L.dxgi == 80 ? 8.0 : 16.0) + (L.files > 1 ? (L.dxgi_b == 80 ? 8.0 : 16.0) : 0.0);
        for (size_t i = 0; i < L.dims.size(); i++) {
            const int w = L.dims[i].first, h = L.dims[i].second;
            const double b = bytes_per_block > 0.0 ? (double)((w + 3) / 4) * ((h + 3) / 4) * bytes_per_block
                                                   : (double)L.stored_C * w * h;
            if (i == 0) { mem_base += b; if (l == 0) mem_base_l0 = b; }
            mem_all += b;
        }
    }
    const double px = (double)std::max(1, g.lat[0].W) * std::max(1, g.lat[0].H) * std::max(1, g.textures_out);
    // Level 0's own share is called out beside the total: it is the number the file layout decides (4 bpp for one BC4,
    // 8 for one BC5, 12 for a BC5 + a BC4 at three channels, 16 for two BC5s), divided by the material's textures.
    const double bpp_base = 8.0 * mem_base / px, bpp_all = 8.0 * mem_all / px, bpp_l0 = 8.0 * mem_base_l0 / px;
    snprintf(l0, sizeof(l0), "Src:%dx%d  L0:%dx%dx%d %s idx%db %dmips  L1:%dx%dx%d %s idx%db %dmips  blk:%d  "
                             "GPU:%.2fbpp/tex base (L0 %.2f), %.2f with mips",
             g.src_w, g.src_h, g.lat[0].W, g.lat[0].H, g.lat[0].C,
             g.lat[0].file_bc ? bcf : (g_bc && g.lat[0].bc_n ? bcs : format_name(g.lat[0].dxgi)),
             g.lat[0].bits[0], g.lat[0].mips, g.lat[1].W, g.lat[1].H, g.lat[1].C, format_name(g.lat[1].dxgi),
             g.lat[1].bits[0], g.lat[1].mips, g.block, bpp_base, bpp_l0, bpp_all);
    // Anisotropy applies in trilinear mode only, so in the other two the strip says the flag is remembered and idle.
    char aniso_s[40];
    if (!g_has_aniso) snprintf(aniso_s, sizeof(aniso_s), "Aniso:OFF (unsupported)");
    else if (!g.aniso) snprintf(aniso_s, sizeof(aniso_s), "Aniso:OFF");
    else if (g.filter_mode == 2) snprintf(aniso_s, sizeof(aniso_s), "Aniso:%u", (unsigned)ANISO_MAX);
    else snprintf(aniso_s, sizeof(aniso_s), "Aniso:%u (trilinear only)", (unsigned)ANISO_MAX);
    const float* const0 = g.cb.scene.const0;
    snprintf(l1, sizeof(l1), "Mode:%-4s Filter:%-9s %-24s Mips:%s L1lod:%s Tex:%d/%d  Show:%s  Renorm:%s",
             g.cube ? "CUBE" : "QUAD", filter_name(g.filter_mode), aniso_s, g.mips_on ? "ON " : "OFF",
             g.lod_bias ? "ON " : "OFF", g.tex_shown, g.textures_out,
             const0[0] > 0.5f ? "LATENT0" : (const0[1] > 0.5f ? "LATENT1" : "DECODE"), const0[3] > 0.5f ? "ON " : "OFF");
    snprintf(l2, sizeof(l2), "X:%+5.1f Y:%+5.1f Z:%5.1f Yaw:%+6.1f Pitch:%+6.1f", (double)g.x, (double)g.y, (double)g.z,
             (double)g.yaw, (double)g.pitch);
    const char* l3 = "F1=Help  Move:Arrows/WS Rot:ADQE C:cube B/T/P:filter X:aniso M:mips L:L1lod N:tex V:renorm 1/2:latents "
                     "4:BC-pack R:reload Spc:reset Esc";
    const char* lines[4] = { l0, l1, l2, l3 };
    for (int li = 0; li < 4; li++) {
        int y = 2 + li * LINE_ADV, x = 4;
        for (const char* p = lines[li]; *p; ++p) { blit_char(buf, x, y, *p); x += 8 * FONT_SCALE; }
    }
    for (int li = 0; li < HELP_N; li++) {
        int y = OVL_H + 6 + li * LINE_ADV, x = 4;
        for (const char* p = HELP_LINES[li]; *p; ++p) { blit_char(buf, x, y, *p); x += 8 * FONT_SCALE; }
    }
    // The decode path, on the state line and at a FIXED column rather than appended to it. The Direct3D viewer
    // has no such state and cannot grow one - it is the program this one is measured against - so the four lines
    // above stay that viewer's text to the pixel and this token sits to the right of where any of them reaches.
    // That is what lets the gate still assert the two strips are byte-identical over the columns they share.
    {
        char coop[16];
        snprintf(coop, sizeof(coop), "Coop:%s", !g_cv_available ? "n-a" : (g_cv_on ? "ON" : "OFF"));
        int x = COOP_COLUMN, y = 2 + LINE_ADV;
        for (const char* p = coop; *p; ++p) { blit_char(buf, x, y, *p); x += 8 * FONT_SCALE; }
    }
    memcpy(g_overlay_mapped, buf.data(), buf.size());
    g_overlay_upload = true;
    g.debug_dirty = false;
}

// The strip's bytes into its image, at the top of the frame's command buffer and only when they have changed. The old
// layout is UNDEFINED the first time, because there is nothing in the image worth preserving, and SHADER_READ_ONLY
// afterwards, because the previous frame sampled it.
static void record_overlay_upload(VkCommandBuffer cmd) {
    if (!g_overlay || !g_overlay_upload) return;
    image_barrier(cmd, g_overlay_image.image, VK_IMAGE_ASPECT_COLOR_BIT,
                  g_overlay_uploaded ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy region{};
    region.bufferOffset = 0; region.bufferRowLength = 0; region.bufferImageHeight = 0;
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { (uint32_t)OVL_W, (uint32_t)OVL_TEX_H, 1 };
    vkCmdCopyBufferToImage(cmd, g_overlay_staging, g_overlay_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    image_barrier(cmd, g_overlay_image.image, VK_IMAGE_ASPECT_COLOR_BIT,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    g_overlay_upload = false;
    g_overlay_uploaded = true;
}

// The strip itself, drawn over the scene: OVL_W x OVL_H pixels of the target, at the top left. The vertices are in
// clip space and are written here rather than at load because they depend on the target's size; with the negative
// viewport height the scene draws under, +1 in y is the TOP, which is what the other viewer's coordinates mean too, so
// the four vertices are that viewer's unchanged.
static void record_overlay_draw(VkCommandBuffer cmd, VkExtent2D extent) {
    if (!g_overlay || !g_overlay_pipeline || !g_overlay_uploaded) return;
    // At 1:1 the strip is OVL_W pixels wide. A window narrower than that would cut it off at the right, so there the
    // strip is scaled down to the window's width, text and all; the --shot frame is never narrower and is unchanged.
    const float fit = extent.width < (uint32_t)OVL_W ? (float)extent.width / (float)OVL_W : 1.0f;
    const int rows = g_help ? OVL_TEX_H : OVL_H;   // the strip alone, or the strip and the help page below it
    const float w = (float)OVL_W * fit / (float)extent.width * 2.0f, h = (float)rows * fit / (float)extent.height * 2.0f;
    const float v = (float)rows / (float)OVL_TEX_H;
    const float verts[16] = {
        -1.0f,     1.0f,     0.0f, 0.0f,
        -1.0f + w, 1.0f,     1.0f, 0.0f,
        -1.0f + w, 1.0f - h, 1.0f, v,
        -1.0f,     1.0f - h, 0.0f, v,
    };
    memcpy(g_overlay_geometry_mapped, verts, sizeof(verts));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_overlay_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_overlay_pipeline_layout, 0, 1, &g_overlay_set, 0, nullptr);
    const VkDeviceSize zero = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &g_overlay_geometry, &zero);
    vkCmdBindIndexBuffer(cmd, g_overlay_geometry, OVERLAY_INDEX_OFFSET, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
}

// ---------------------------------------------------------------------------
// Geometry: the quad and the cube, the Direct3D viewer's vertices verbatim, in ONE host-visible buffer at four
// offsets. No staging copy and no device-local buffer, because at a few hundred bytes it does not matter and the
// honest simple thing reads better.
// ---------------------------------------------------------------------------
static VkBuffer       g_geometry = VK_NULL_HANDLE;
static VkDeviceMemory g_geometry_memory = VK_NULL_HANDLE;
static VkDeviceSize   g_quad_vertices = 0, g_quad_indices = 0, g_cube_vertices = 0, g_cube_indices = 0;
static uint32_t       g_cube_index_count = 0;

static void create_geometry(float aspect) {
    float hw, hh;
    if (aspect >= 1.0f) { hw = 1.0f; hh = 1.0f / aspect; } else { hw = aspect; hh = 1.0f; }
    // UV origin: (0, 0) = top-left = the first uploaded row; the .dds levels are top-row-first, so v runs 0..1 top to
    // bottom with no flip - in either api, because the flip Vulkan needs is in the viewport and not in the data.
    const float quad_v[] = {
        -hw, -hh, 0.0f, 0.0f, 1.0f,   hw, -hh, 0.0f, 1.0f, 1.0f,
         hw,  hh, 0.0f, 1.0f, 0.0f,  -hw,  hh, 0.0f, 0.0f, 0.0f,
    };
    const uint32_t quad_i[] = { 0, 1, 2, 0, 2, 3 };
    const float h = 0.5f;
    const float cube_v[] = {
        -h, -h,  h, 0, 1,   h, -h,  h, 1, 1,   h,  h,  h, 1, 0,  -h,  h,  h, 0, 0,
         h, -h, -h, 0, 1,  -h, -h, -h, 1, 1,  -h,  h, -h, 1, 0,   h,  h, -h, 0, 0,
         h, -h,  h, 0, 1,   h, -h, -h, 1, 1,   h,  h, -h, 1, 0,   h,  h,  h, 0, 0,
        -h, -h, -h, 0, 1,  -h, -h,  h, 1, 1,  -h,  h,  h, 1, 0,  -h,  h, -h, 0, 0,
        -h,  h,  h, 0, 1,   h,  h,  h, 1, 1,   h,  h, -h, 1, 0,  -h,  h, -h, 0, 0,
        -h, -h, -h, 0, 1,   h, -h, -h, 1, 1,   h, -h,  h, 1, 0,  -h, -h,  h, 0, 0,
    };
    std::vector<uint32_t> cube_i;
    for (uint32_t f = 0; f < 6; f++) {
        const uint32_t b = f * 4;
        const uint32_t six[6] = { b, b + 1, b + 2, b, b + 2, b + 3 };
        cube_i.insert(cube_i.end(), six, six + 6);
    }
    g_cube_index_count = (uint32_t)cube_i.size();

    g_quad_vertices = 0;
    g_quad_indices = g_quad_vertices + sizeof(quad_v);
    g_cube_vertices = g_quad_indices + sizeof(quad_i);
    g_cube_indices = g_cube_vertices + sizeof(cube_v);
    const VkDeviceSize total = g_cube_indices + cube_i.size() * sizeof(uint32_t);
    create_buffer(total, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  g_geometry, g_geometry_memory);
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(g_device, g_geometry_memory, 0, total, 0, &mapped));
    memcpy((uint8_t*)mapped + g_quad_vertices, quad_v, sizeof(quad_v));
    memcpy((uint8_t*)mapped + g_quad_indices, quad_i, sizeof(quad_i));
    memcpy((uint8_t*)mapped + g_cube_vertices, cube_v, sizeof(cube_v));
    memcpy((uint8_t*)mapped + g_cube_indices, cube_i.data(), cube_i.size() * sizeof(uint32_t));
    vkUnmapMemory(g_device, g_geometry_memory);
}

// The scene constants, rebuilt each frame from the camera and the target's own size. The matrix is transposed once
// here, exactly as the Direct3D viewer transposes it at constant-buffer-write time, so the two send the same bytes.
static void set_uniforms(VkExtent2D extent) {
    const Mat4 proj = mat_perspective(FOV_DEGREES, (float)extent.width / (float)extent.height, 0.001f, 100.0f);
    const Mat4 model = mat_mul(mat_mul(mat_translate(g.x, g.y, g.z), mat_rot_y(g.yaw)), mat_rot_x(g.pitch));
    const Mat4 mvp_t = mat_transpose(mat_mul(proj, model));
    SceneConstants& s = g.cb.scene;
    memcpy(s.mvp, mvp_t.m, sizeof(s.mvp));
    s.tex_size[0] = (float)g.lat[0].W; s.tex_size[1] = (float)g.lat[0].H;
    s.tex_size[2] = (float)g.lat[1].W; s.tex_size[3] = (float)g.lat[1].H;
    s.lod_info[0] = (float)(g.lat[0].mips - 1); s.lod_info[1] = (float)(g.lat[1].mips - 1);
    // const0.z: the factor the shader multiplies level 1's UV derivatives by before its textureGrad.
    // 2^lod_bias_level1 raises the hardware's LOD by lod_bias_level1 mips, which is the encoder's 1:1 rule; key L off
    // leaves the gradients alone, which is the control. It is written here every frame, so no key and no reset can
    // leave a zero scale in the slot - which would collapse every fetch to mip 0.
    s.const0[2] = g.lod_bias ? std::exp2(g.lod_bias_level1) : 1.0f;
    g.cb.dec.sel[1] = g.tex_shown;
    // The textures level 0 is read from, decided from the SAME state update_descriptors binds from, so that the count
    // the shader is told and the textures it is given can never be one frame apart - which on a key-4 toggle of a
    // three- or four-channel level 0 would be a visible flash of the wrong plane.
    const bool packed = g_bc && g.lat[0].bc_n > 0;
    g.cb.dec.sel[2] = packed ? g.lat[0].bc_n : (g.lat[0].files > 1 ? 2 : 0);
    memcpy(g_uniform_mapped, &g.cb, sizeof(Uniforms));
}

// The frame itself, which is the same recording whether the target is a swapchain image or the offscreen image of
// --shot. The two paths share everything except the image they are given.
static void record_scene(VkCommandBuffer cmd, VkImageView colour_view, VkExtent2D extent) {
    // The overlay's text is rebuilt on the host and copied into its image BEFORE the rendering begins: a copy cannot
    // be recorded inside a dynamic-rendering block.
    update_debug_text();
    record_overlay_upload(cmd);

    VkRenderingAttachmentInfo colour{};
    colour.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colour.imageView = colour_view;
    colour.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colour.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colour.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    memcpy(colour.clearValue.color.float32, CLEAR_COLOUR, sizeof(CLEAR_COLOUR));

    VkRenderingAttachmentInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth.imageView = g_depth_view;
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.clearValue.depthStencil.depth = 1.0f;

    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.offset = { 0, 0 };
    ri.renderArea.extent = extent;
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1; ri.pColorAttachments = &colour;
    ri.pDepthAttachment = &depth;
    vkCmdBeginRendering(cmd, &ri);

    // The y flip, and the only place it happens: Vulkan's clip space has y pointing down where Direct3D's points up,
    // and a negative viewport height (core since 1.1) pays for that in one line rather than in the projection matrix -
    // which is what lets the two viewers share a camera that can be compared parameter for parameter.
    VkViewport viewport{};
    viewport.x = 0.0f; viewport.y = (float)extent.height;
    viewport.width = (float)extent.width; viewport.height = -(float)extent.height;
    viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{};
    scissor.offset = { 0, 0 }; scissor.extent = extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Which of the two decode paths this frame draws with, and the ONLY place that choice is made. The two
    // pipelines differ in their fragment shader and in nothing else, so the switch is one handle.
    // The two timestamps --bench reads, and they are around THE SCENE DRAW rather than around the frame: the
    // overlay's upload and its own draw are not what the two decode paths differ in, and a measurement that
    // included them would be measuring a constant along with the thing being compared. ALL_COMMANDS on both ends
    // is what makes the pair an interval that contains the draw and nothing of what follows it.
    if (g_query_pool) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, g_query_pool, 0);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_cv_on ? g_pipeline_cv : g_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipeline_layout, 0, 1, &g_descriptor_set, 0, nullptr);
    const VkDeviceSize vertices = g.cube ? g_cube_vertices : g_quad_vertices;
    vkCmdBindVertexBuffers(cmd, 0, 1, &g_geometry, &vertices);
    vkCmdBindIndexBuffer(cmd, g_geometry, g.cube ? g_cube_indices : g_quad_indices, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, g.cube ? g_cube_index_count : 6u, 1, 0, 0, 0);
    if (g_query_pool) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, g_query_pool, 1);

    record_overlay_draw(cmd, extent);   // the debug strip over the scene, unless --nooverlay

    vkCmdEndRendering(cmd);
}

// The depth image's contents are cleared at the start of every frame, so it is transitioned from UNDEFINED each time:
// there is nothing in it worth preserving and saying so lets the driver take the cheaper path.
static void barrier_depth_for_rendering(VkCommandBuffer cmd) {
    image_barrier(cmd, g_depth_image, VK_IMAGE_ASPECT_DEPTH_BIT,
                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
}

// Whether the surface's own idea of its size has moved away from the swapchain's, which is the ONE thing a recreate
// can fix and therefore the test a SUBOPTIMAL is acted on by.
//
// SUBOPTIMAL means "this still presents, but not perfectly". At a resize it arrives once and a recreate settles it.
// But nothing in the spec says it has to stop: a driver presenting through a scaler or a rotation can report it on
// every frame for as long as that arrangement lasts, and recreating on each one would mean a vkDeviceWaitIdle, a
// teardown and a rebuild every frame - a viewer that runs at single-figure frame rates for a reason the user cannot
// see. So the extent decides. A surface that leaves the size to the application (currentExtent 0xFFFFFFFF) reports
// nothing to compare, and there the window's own resize message is what drives a recreate, as it always was.
static bool surface_extent_changed(void) {
    VkSurfaceCapabilitiesKHR caps{};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_phys, g_surface, &caps) != VK_SUCCESS) return false;
    if (caps.currentExtent.width == 0xFFFFFFFFu) return false;
    return caps.currentExtent.width != g_swapchain_extent.width ||
           caps.currentExtent.height != g_swapchain_extent.height;
}

static bool recreate_swapchain(void) {
    VK_CHECK(vkDeviceWaitIdle(g_device));
    destroy_swapchain_objects();
    if (!create_swapchain()) return false;
    // A surface whose format changed under us would leave the pipeline declaring the wrong attachment format. It does
    // not happen in practice, and it costs one comparison to be sure rather than to assume.
    if (g_swapchain_format != g_pipeline_format && !create_pipeline(g_swapchain_format)) exit(1);
    return true;
}

// Everything the asset, the samplers, the descriptors and the geometry need, in the order they need each other. It is
// one function because both the windowed path and the headless one do exactly this, and a difference between them
// would be a difference between two frames that are supposed to be the same.
static bool load_everything(const std::string& json_path, const char* argv0) {
    create_descriptors();
    create_dummy_image();
    if (!load_asset(json_path)) return false;
    // --tex is given before the asset is read, so the material's texture count is only known now; an out-of-range
    // value would index the shader's output triples past the end.
    if (g.tex_shown < 0 || g.tex_shown >= g.textures_out) {
        fprintf(stderr, "WARNING: --tex %d is outside 0..%d; showing texture 0\n", g.tex_shown, g.textures_out - 1);
        g.tex_shown = 0;
    }
    update_descriptors();
#if NNTC_HAVE_COOPVEC
    if (!build_cooperative_vector_weights()) return false;
#endif
    create_geometry((float)g.lat[0].W / (float)g.lat[0].H);
    init_overlay();
    g_vert_path = shader_path(argv0, "view.vert");
    g_frag_path = shader_path(argv0, "view.frag");
    g_frag_cv_path = shader_path(argv0, "view_coopvec.frag");
    return true;
}

// ---------------------------------------------------------------------------
// The keys. Every one of them is the other viewer's letter with the other viewer's meaning, and the line each prints
// is that viewer's line, because a reader comparing the two programs' consoles is comparing the two APIs and should
// not have to allow for two ways of saying the same thing.
//
// What a key actually changes is one of three things: a flag the next frame's uniform block carries (1, 2, V, N, C,
// the camera), a sampler or an image the next frame's descriptors point at (P, B, T, X, M, L, 4), or the pipeline
// itself (R). None of them touches anything mid-frame: the loop waits on the fence before it records, so every one of
// these is a plain write.
// ---------------------------------------------------------------------------
static void reload_shaders(void) {
    // The old pipeline may still be in flight, and create_pipeline destroys it on success, so the device is drained
    // first. A shader that does not compile leaves the old pipeline exactly where it was and says so, which is the
    // whole point of the key: an edit that will not compile must not take the picture away.
    VK_CHECK(vkDeviceWaitIdle(g_device));
    if (!create_pipeline(g_pipeline_format))
        fprintf(stderr, "Shader reload failed, keeping previous shader.\n");
}

bool app_help_open(void) { return g_help; }

void app_key_struck(int key) {
    float* const0 = g.cb.scene.const0;
    float* const1 = g.cb.scene.const1;
    // With the help page open any key closes it and does nothing else: Esc closes the page, not the viewer, and a
    // held movement key that closed it does not move the camera until it is released. KEY_OTHER is no key the
    // held-key scan reads, so there is nothing to wait for.
    if (g_help) { g_help = false; g_help_eat = key == KEY_OTHER ? 0 : key; g.debug_dirty = true; return; }
    switch (key) {
        case KEY_F1: g_help = true; g.debug_dirty = true; break;
        case KEY_ESCAPE: g_quit_requested = true; break;
        case 'R': reload_shaders(); break;
        case 'B': g.filter_mode = 1; g.debug_dirty = true; printf("Filter: BILINEAR\n"); break;
        case 'T': g.filter_mode = 2; g.debug_dirty = true; printf("Filter: TRILINEAR\n"); break;
        case 'P': g.filter_mode = 0; g.debug_dirty = true; printf("Filter: POINT\n"); break;
        case 'X':
            // A device without samplerAnisotropy has no anisotropic sampler to switch to - every one of the eight
            // was created with anisotropyEnable false - so the key says so and changes nothing, rather than flipping a
            // flag the overlay would then report as an anisotropy that is not happening.
            if (!g_has_aniso) { printf("anisotropy unsupported on this device\n"); break; }
            g.aniso = !g.aniso; g.debug_dirty = true;
            printf("Anisotropic filtering: %s%s\n", g.aniso ? "ON (MaxAnisotropy " : "OFF",
                   g.aniso ? (std::to_string((unsigned)ANISO_MAX) + ")").c_str() : "");
            if (g.filter_mode != 2) printf("  (it applies in TRILINEAR mode only; press T)\n");
            break;
        case 'L':
            g.lod_bias = !g.lod_bias; g.debug_dirty = true;
            printf("Level 1 LOD shift: %s\n", g.lod_bias ? "ON (gradients scaled by 2^lod_bias_level1: the 1:1 mip rule)"
                                                         : "OFF (the GPU's own LOD: level 1 two mips finer than fitted)");
            break;
        case 'M':
            g.mips_on = !g.mips_on; g.debug_dirty = true;
            printf("Mips: %s\n", g.mips_on ? "ON (MaxLOD unlimited)" : "OFF (sampler MaxLOD = 0: mip 0 only)");
            break;
        case 'C': g.cube = !g.cube; g.debug_dirty = true; break;
        case 'K':
            // Cooperative vectors on or off. It is a different PIPELINE and not a different constant, and the key
            // exists so that the comparison of the two decode paths is a keystroke rather than a rebuild - which
            // is what makes it likely to be made. On a device without the extension it says so and stays plain.
            if (!g_cv_available) { printf("decode: plain (%s)\n", g_cv_why_not.c_str()); break; }
            g_cv_on = !g_cv_on; g.debug_dirty = true;
            printf("decode: %s\n", g_cv_on ? "cooperative vectors" : "plain");
            break;
        case 'N':
            g.tex_shown = (g.tex_shown + 1) % g.textures_out; g.debug_dirty = true;
            printf("Output texture %d of %d\n", g.tex_shown, g.textures_out);
            break;
        case '1': const0[0] = 1.0f - const0[0]; const0[1] = 0.0f; g.debug_dirty = true; break;
        case '2': const0[1] = 1.0f - const0[1]; const0[0] = 0.0f; g.debug_dirty = true; break;
        case 'V':
            const0[3] = 1.0f - const0[3]; g.debug_dirty = true;
            printf("Normal renormalisation: %s\n", const0[3] > 0.5f ? "ON (unpack, unit length, repack)" : "OFF");
            break;
        case '4':
            // Nothing at all on a file whose level 0 is already block-compressed: there is no pack to switch to, which
            // is what the other viewer does and what viewer/README.md says.
            if (g.lat[0].bc_n) {
                g_bc = !g_bc; g.debug_dirty = true;
                printf("Level 0: %s\n", g_bc ? "BC4/BC5 pack" : "uncompressed");
            }
            break;
        case '5': const1[0] = 1.0f - const1[0]; g.debug_dirty = true; break;
        case '6': const1[1] = 1.0f - const1[1]; g.debug_dirty = true; break;
        case '7': const1[2] = 1.0f - const1[2]; g.debug_dirty = true; break;
        case '8': const1[3] = 1.0f - const1[3]; g.debug_dirty = true; break;
        case KEY_SPACE:
            g.x = 0; g.y = 0; g.z = -3.0f; g.yaw = 0; g.pitch = 0;
            for (int i = 0; i < 4; i++) { const0[i] = 0; const1[i] = 0; }
            g.tex_shown = 0; g.debug_dirty = true;
            printf("Reset to initial state\n");
            break;
        default: break;
    }
}

static void app_process_held_keys(float dt) {
    if (g_shot) return;   // the shot frame is a function of the command line and the asset, and of nothing else
    if (g_help) return;   // the help page is modal: nothing moves while it is open
    if (g_help_eat && !win_key_held(g_help_eat)) g_help_eat = 0;   // the key that closed the page is up again
    auto held = [](int key) { return key != g_help_eat && win_key_held(key); };
    if (held(KEY_SHIFT)) dt *= 1.0f / 3.0f;
    bool moved = false;
    if (held('W')) { g.z += Z_SPEED * dt; moved = true; }
    if (held('S')) { g.z -= Z_SPEED * dt; moved = true; }
    if (held(KEY_LEFT))  { g.x += XY_SPEED * dt; moved = true; }
    if (held(KEY_RIGHT)) { g.x -= XY_SPEED * dt; moved = true; }
    if (held(KEY_UP))    { g.y += XY_SPEED * dt; moved = true; }
    if (held(KEY_DOWN))  { g.y -= XY_SPEED * dt; moved = true; }
    if (held('A')) { g.yaw += ROT_SPEED * dt; moved = true; }
    if (held('D')) { g.yaw -= ROT_SPEED * dt; moved = true; }
    if (held('Q')) { g.pitch += ROT_SPEED * dt; moved = true; }
    if (held('E')) { g.pitch -= ROT_SPEED * dt; moved = true; }
    if (g.z < Z_MAX) g.z = Z_MAX;
    if (g.z > Z_MIN) g.z = Z_MIN;
    if (moved) g.debug_dirty = true;
}

// ---------------------------------------------------------------------------
// The 24-bit bottom-up .bmp writer, byte for byte the Direct3D viewer's, so that the two programs' shots can be
// compared with nothing between them. A shot that was not written is a failed run with the exit code to say so: a
// script that checks a screenshot must not be told the frame exists when it does not.
// ---------------------------------------------------------------------------
static bool write_bmp(const char* path, const uint8_t* rgba, int W, int H, size_t row_pitch) {
    const int pitch = (W * 3 + 3) & ~3;
    std::vector<uint8_t> out((size_t)pitch * (size_t)H, 0);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const uint8_t* sp = rgba + (size_t)y * row_pitch + (size_t)x * 4;
            uint8_t* op = &out[(size_t)(H - 1 - y) * (size_t)pitch + (size_t)x * 3];
            op[0] = sp[2]; op[1] = sp[1]; op[2] = sp[0];
        }
    const uint32_t fsz = 54 + (uint32_t)out.size(); uint8_t hdr[54] = { 'B', 'M' }; uint32_t v;
    v = fsz; memcpy(hdr + 2, &v, 4); v = 54; memcpy(hdr + 10, &v, 4); v = 40; memcpy(hdr + 14, &v, 4);
    v = (uint32_t)W; memcpy(hdr + 18, &v, 4); v = (uint32_t)H; memcpy(hdr + 22, &v, 4);
    hdr[26] = 1; hdr[28] = 24; v = (uint32_t)out.size(); memcpy(hdr + 34, &v, 4);
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", path); return false; }
    const bool ok = fwrite(hdr, 1, 54, f) == 54 && fwrite(out.data(), 1, out.size(), f) == out.size();
    if (fclose(f) != 0 || !ok) { fprintf(stderr, "ERROR: writing %s failed\n", path); return false; }
    printf("wrote %s (%dx%d)\n", path, W, H);
    return true;
}

// ---------------------------------------------------------------------------
// --bench N: N headless frames, timed on the GPU.
//
// Wall-clock frame time is useless for this program and the reason is structural rather than a matter of care: the
// frame loop waits on the device at the end of every frame by design, so a clock on the host measures the wait. What
// is measured instead is the interval between two timestamps written into the command buffer either side of the scene
// draw, turned into nanoseconds by the device's own timestampPeriod. A queue family that reports timestampValidBits 0
// cannot be timed at all, and that is a refusal rather than a number made out of nothing.
//
// The mean and the MINIMUM are both printed. The minimum is the honest floor - the frame that met the least
// interference from everything else on the device - and the mean is what a reader would otherwise compute wrongly
// from one number. Neither is estimated: both come out of the query pool.
// ---------------------------------------------------------------------------
static int run_bench(int frames, uint32_t W, uint32_t H) {
    if (!g_timestamp_bits) {
        fprintf(stderr, "ERROR: --bench: this device's queue family reports timestampValidBits 0, so a frame here "
                        "cannot be timed\n");
        return 1;
    }
    VkQueryPoolCreateInfo qi{};
    qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP; qi.queryCount = 2;
    VK_CHECK(vkCreateQueryPool(g_device, &qi, nullptr, &g_query_pool));

    set_uniforms({ W, H });
    const uint64_t mask = g_timestamp_bits >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << g_timestamp_bits) - 1);
    double total = 0.0, best = 0.0;
    for (int i = 0; i < frames; i++) {
        VK_CHECK(vkResetCommandBuffer(g_command_buffer, 0));
        VkCommandBufferBeginInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(g_command_buffer, &cbi));
        // The reset has to be here rather than beside the writes: a query pool cannot be reset inside a rendering
        // block, and the writes themselves are inside one.
        vkCmdResetQueryPool(g_command_buffer, g_query_pool, 0, 2);
        image_barrier(g_command_buffer, g_shot_target.image, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        barrier_depth_for_rendering(g_command_buffer);
        record_scene(g_command_buffer, g_shot_target.view, { W, H });
        VK_CHECK(vkEndCommandBuffer(g_command_buffer));

        VkCommandBufferSubmitInfo csi{};
        csi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        csi.commandBuffer = g_command_buffer;
        VkSubmitInfo2 si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        si.commandBufferInfoCount = 1; si.pCommandBufferInfos = &csi;
        VK_CHECK(vkQueueSubmit2(g_queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(g_queue));

        uint64_t t[2] = { 0, 0 };
        VK_CHECK(vkGetQueryPoolResults(g_device, g_query_pool, 0, 2, sizeof(t), t, sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        // Only the valid bits of the counter mean anything, and it wraps within them, so the difference is taken
        // inside the mask rather than on the raw values.
        const uint64_t ticks = ((t[1] & mask) - (t[0] & mask)) & mask;
        const double us = (double)ticks * (double)g_timestamp_period / 1000.0;
        total += us;
        if (i == 0 || us < best) best = us;
    }
    printf("bench: %d frames at %ux%u, decode %s: mean %.1f us, min %.1f us per scene draw\n", frames, W, H,
           g_cv_on ? "cooperative vectors" : "plain", total / (double)frames, best);
    vkDestroyQueryPool(g_device, g_query_pool, nullptr);
    g_query_pool = VK_NULL_HANDLE;
    return 0;
}

// ---------------------------------------------------------------------------
// --shot: one frame, headless. No window, no surface, no swapchain and no presentation engine - an offscreen image, the
// same recording, and a copy back through a staging buffer. The frame is then a function of the command line alone for
// a stronger reason than the Direct3D viewer's: there is no window to receive a keystroke and no compositor to
// interact with.
// ---------------------------------------------------------------------------
static int render_shot(const std::string& path, const std::string& json_path, int device_index, const char* argv0,
                       int bench) {
    create_instance(false);
    pick_physical_device(device_index);
    check_target_size();
    create_device(false);
    create_command_objects();
    if (!load_everything(json_path, argv0)) return 1;

    const uint32_t W = (uint32_t)FRAME_WIDTH, H = (uint32_t)FRAME_HEIGHT;
    VkImageCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D; ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent = { W, H, 1 }; ii.mipLevels = 1; ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE; ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // The offscreen target lives in a file-scope GpuImage rather than in three locals, so that destroy_everything
    // can reach it: a shader that will not compile returns from the middle of this function, and the three objects
    // would otherwise still be alive when the device is destroyed - which is a leak the Debug gate greps for.
    VK_CHECK(vkCreateImage(g_device, &ii, nullptr, &g_shot_target.image));
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g_device, g_shot_target.image, &req);
    VkMemoryAllocateInfo ma{};
    ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ma.allocationSize = req.size;
    ma.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(g_device, &ma, nullptr, &g_shot_target.memory));
    VK_CHECK(vkBindImageMemory(g_device, g_shot_target.image, g_shot_target.memory, 0));
    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = g_shot_target.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = VK_FORMAT_R8G8B8A8_UNORM;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VK_CHECK(vkCreateImageView(g_device, &vi, nullptr, &g_shot_target.view));
    create_depth(W, H);
    if (!create_pipeline(VK_FORMAT_R8G8B8A8_UNORM)) return 1;

    // --bench renders into the same offscreen image through the same recording and writes no file: what it
    // measures has to be the frame --shot would have written, or the number is about something else.
    if (bench > 0) {
        const int status = run_bench(bench, W, H);
        destroy_depth();
        destroy_image(g_shot_target);
        return status;
    }

    // The staging buffer the frame is copied into: host visible and coherent, so the map below needs no invalidate,
    // and tightly packed because bufferRowLength = 0 in the copy says the rows are W texels of four bytes.
    //
    // HOST_CACHED as well, WHERE THE DEVICE HAS IT, because this buffer is the one this program READS. An uncached
    // host-visible allocation on a discrete card is write-combined: writes stream, but reads come back uncached over
    // the bus a few bytes at a time, and write_bmp walks every byte of a 2560x1440 frame - about 15 MB - in that
    // state. The upload staging elsewhere in this file is written and never read, so it wants the opposite and is
    // left alone. COHERENT is kept in the preferred set so the mapping still needs no vkInvalidateMappedMemoryRanges;
    // a device with no cached+coherent type falls back to what this always used.
    const VkDeviceSize bytes = (VkDeviceSize)W * H * 4;
    VkBuffer staging = VK_NULL_HANDLE; VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    create_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, staging_memory,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                  VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

    set_uniforms({ W, H });
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(g_command_buffer, &cbi));
    image_barrier(g_command_buffer, g_shot_target.image, VK_IMAGE_ASPECT_COLOR_BIT,
                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    barrier_depth_for_rendering(g_command_buffer);
    record_scene(g_command_buffer, g_shot_target.view, { W, H });
    image_barrier(g_command_buffer, g_shot_target.image, VK_IMAGE_ASPECT_COLOR_BIT,
                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy region{};
    region.bufferOffset = 0; region.bufferRowLength = 0; region.bufferImageHeight = 0;
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { W, H, 1 };
    vkCmdCopyImageToBuffer(g_command_buffer, g_shot_target.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1,
                           &region);
    VK_CHECK(vkEndCommandBuffer(g_command_buffer));

    VkCommandBufferSubmitInfo csi{};
    csi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    csi.commandBuffer = g_command_buffer;
    VkSubmitInfo2 si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si.commandBufferInfoCount = 1; si.pCommandBufferInfos = &csi;
    VK_CHECK(vkQueueSubmit2(g_queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(g_queue));

    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(g_device, staging_memory, 0, bytes, 0, &mapped));
    const bool ok = write_bmp(path.c_str(), (const uint8_t*)mapped, (int)W, (int)H, (size_t)W * 4);
    vkUnmapMemory(g_device, staging_memory);

    vkDestroyBuffer(g_device, staging, nullptr);
    vkFreeMemory(g_device, staging_memory, nullptr);
    destroy_depth();
    destroy_image(g_shot_target);
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// The interactive path: a window, a swapchain, and the frame loop.
//
// ONE FRAME IN FLIGHT. The fence is waited on before anything of the next frame is recorded, so the device is idle
// with respect to this program's own work at that point and every buffer, descriptor and image can be rewritten in
// place. That is a deliberate simplification and the right one for a viewer: it costs the overlap of cpu and gpu, and
// it removes the single most error-prone part of a first Vulkan program.
// ---------------------------------------------------------------------------
static int render_window(const std::string& json_path, int device_index, const char* argv0) {
    if (!win_open(WINDOW_WIDTH, WINDOW_HEIGHT, "NNTC latent viewer (Vulkan)")) return 1;
    win_client_size(&WINDOW_WIDTH, &WINDOW_HEIGHT);
    printf("window: %dx%d client area\n", WINDOW_WIDTH, WINDOW_HEIGHT);

    create_instance(true);
    g_surface = win_create_surface(g_instance);
    if (g_surface == VK_NULL_HANDLE) return 1;
    pick_physical_device(device_index);
    check_target_size();
    create_device(true);
    create_command_objects();
    if (!load_everything(json_path, argv0)) return 1;
    if (!create_swapchain()) { fprintf(stderr, "ERROR: the window has no area to draw in\n"); return 1; }
    printf("swapchain: %ux%u, %u images, format %d, FIFO\n", g_swapchain_extent.width, g_swapchain_extent.height,
           (unsigned)g_swapchain_images.size(), (int)g_swapchain_format);
    if (!create_pipeline(g_swapchain_format)) return 1;
    // Everything that can refuse has now not: the device query, --coopvec 1 against it, the asset, and both shaders.
    // Only here does the window reach the screen, so a refused run never flashes one up.
    win_show();

    VkSemaphoreCreateInfo sem{};
    sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VK_CHECK(vkCreateSemaphore(g_device, &sem, nullptr, &g_image_available));
    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;   // the first frame has nothing to wait for
    VK_CHECK(vkCreateFence(g_device, &fi, nullptr, &g_frame_fence));

    const double started = win_seconds();
    double last = started;
    uint64_t frames = 0;
    while (!win_should_quit()) {
        win_pump();
        if (win_should_quit()) break;

        // The frame's own time, clamped as the other viewer clamps it, so that a stall does not fling the camera.
        const double now = win_seconds();
        float dt = (float)(now - last);
        last = now;
        if (dt > 0.1f) dt = 0.1f;
        app_process_held_keys(dt);

        VK_CHECK(vkWaitForFences(g_device, 1, &g_frame_fence, VK_TRUE, UINT64_MAX));

        if (g_resized) {
            g_resized = false;
            recreate_swapchain();
        }

        // THERE IS NO SWAPCHAIN while the window is minimised or has been dragged down to no client area at all:
        // recreate_swapchain destroyed the old one and create_swapchain refused to make a new one of zero extent. So
        // the loop waits for the window system rather than acquiring on a null handle, which is a segfault on the very
        // next call, and tries again - a restore or a resize posts WM_SIZE, which is exactly what win_idle waits for.
        // Quitting still works from here, because WM_CLOSE wakes it too.
        if (g_swapchain == VK_NULL_HANDLE) {
            win_idle();
            continue;
        }

        uint32_t image = 0;
        VkResult acquired = vkAcquireNextImageKHR(g_device, g_swapchain, UINT64_MAX, g_image_available,
                                                  VK_NULL_HANDLE, &image);
        // Both of these mean "this swapchain no longer matches the surface", and both are handled here and at the
        // present below: OUT_OF_DATE cannot be drawn with at all, SUBOPTIMAL can be drawn with and is rebuilt only
        // when the surface's extent has actually changed under it (surface_extent_changed says why), which is what
        // keeps a resize from dropping a frame without letting a standing SUBOPTIMAL rebuild the swapchain forever.
        if (acquired == VK_ERROR_OUT_OF_DATE_KHR) { recreate_swapchain(); continue; }
        if (acquired == VK_SUBOPTIMAL_KHR && surface_extent_changed()) g_resized = true;
        if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) {
            fprintf(stderr, "ERROR: vkAcquireNextImageKHR returned %s (%d)\n", vk_result_name(acquired),
                    (int)acquired);
            return 1;
        }

        // The frame's bindings are decided here, after the fence: keys P/B/T, X, M and 4 change which sampler and
        // which image the descriptors point at, and rewriting them with the device idle with respect to this
        // program's own work is what one frame in flight buys.
        update_descriptors();
        set_uniforms(g_swapchain_extent);
        VK_CHECK(vkResetFences(g_device, 1, &g_frame_fence));
        VK_CHECK(vkResetCommandBuffer(g_command_buffer, 0));
        VkCommandBufferBeginInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(g_command_buffer, &cbi));
        image_barrier(g_command_buffer, g_swapchain_images[image], VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        barrier_depth_for_rendering(g_command_buffer);
        record_scene(g_command_buffer, g_swapchain_views[image], g_swapchain_extent);
        image_barrier(g_command_buffer, g_swapchain_images[image], VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
        VK_CHECK(vkEndCommandBuffer(g_command_buffer));

        VkCommandBufferSubmitInfo csi{};
        csi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        csi.commandBuffer = g_command_buffer;
        VkSemaphoreSubmitInfo wait{};
        wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        wait.semaphore = g_image_available;
        wait.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSemaphoreSubmitInfo signal{};
        signal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        signal.semaphore = g_render_finished[image];
        signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkSubmitInfo2 si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        si.waitSemaphoreInfoCount = 1; si.pWaitSemaphoreInfos = &wait;
        si.commandBufferInfoCount = 1; si.pCommandBufferInfos = &csi;
        si.signalSemaphoreInfoCount = 1; si.pSignalSemaphoreInfos = &signal;
        VK_CHECK(vkQueueSubmit2(g_queue, 1, &si, g_frame_fence));

        VkPresentInfoKHR pi{};
        pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = &g_render_finished[image];
        pi.swapchainCount = 1; pi.pSwapchains = &g_swapchain; pi.pImageIndices = &image;
        const VkResult presented = vkQueuePresentKHR(g_queue, &pi);
        if (presented == VK_ERROR_OUT_OF_DATE_KHR) g_resized = true;
        else if (presented == VK_SUBOPTIMAL_KHR) { if (surface_extent_changed()) g_resized = true; }
        else if (presented != VK_SUCCESS) {
            fprintf(stderr, "ERROR: vkQueuePresentKHR returned %s (%d)\n", vk_result_name(presented),
                    (int)presented);
            return 1;
        }
        frames++;
    }

    VK_CHECK(vkDeviceWaitIdle(g_device));
    printf("%llu frames in %.1f s\n", (unsigned long long)frames, win_seconds() - started);
    vkDestroyFence(g_device, g_frame_fence, nullptr);
    vkDestroySemaphore(g_device, g_image_available, nullptr);
    destroy_swapchain_objects();
    return 0;
}

// ---------------------------------------------------------------------------
// Everything this program created, destroyed on the way out, in the order that respects what depends on what:
// pipelines before their layouts, views before their images, memory after what was bound to it, the device after
// everything inside it and the instance last of all.
//
// The operating system would free all of it anyway - a process that exits gives back its own memory and the driver's -
// and it is done explicitly for one reason: the validation layer reports every object still alive when the device is
// destroyed, so a Debug run that says nothing at exit is the check that this program owns what it makes. Every handle
// is tested, so a path that exited early destroys exactly what it had created.
// ---------------------------------------------------------------------------
static void destroy_everything(void) {
    if (g_device) {
        // The queue may still be working through the last frame; nothing below may be destroyed while it is in use.
        // A device that has been lost returns an error here and there is nothing to do about it at this point, so the
        // result is deliberately not checked: the teardown carries on and the exit code stays the frame loop's.
        vkDeviceWaitIdle(g_device);

        if (g_pipeline) { vkDestroyPipeline(g_device, g_pipeline, nullptr); g_pipeline = VK_NULL_HANDLE; }
        if (g_pipeline_cv) { vkDestroyPipeline(g_device, g_pipeline_cv, nullptr); g_pipeline_cv = VK_NULL_HANDLE; }
        if (g_overlay_pipeline) { vkDestroyPipeline(g_device, g_overlay_pipeline, nullptr); g_overlay_pipeline = VK_NULL_HANDLE; }
        if (g_pipeline_layout) { vkDestroyPipelineLayout(g_device, g_pipeline_layout, nullptr); g_pipeline_layout = VK_NULL_HANDLE; }
        if (g_overlay_pipeline_layout) { vkDestroyPipelineLayout(g_device, g_overlay_pipeline_layout, nullptr); g_overlay_pipeline_layout = VK_NULL_HANDLE; }

        // A pool's sets go with it, so the sets themselves are not freed one by one.
        if (g_descriptor_pool) { vkDestroyDescriptorPool(g_device, g_descriptor_pool, nullptr); g_descriptor_pool = VK_NULL_HANDLE; }
        if (g_overlay_pool) { vkDestroyDescriptorPool(g_device, g_overlay_pool, nullptr); g_overlay_pool = VK_NULL_HANDLE; }
        if (g_set_layout) { vkDestroyDescriptorSetLayout(g_device, g_set_layout, nullptr); g_set_layout = VK_NULL_HANDLE; }
        if (g_overlay_set_layout) { vkDestroyDescriptorSetLayout(g_device, g_overlay_set_layout, nullptr); g_overlay_set_layout = VK_NULL_HANDLE; }

        for (int mips = 0; mips < 2; mips++) for (int f = 0; f < FILTER_STATES; f++) {
            if (g_samplers[mips][f]) { vkDestroySampler(g_device, g_samplers[mips][f], nullptr); g_samplers[mips][f] = VK_NULL_HANDLE; }
        }
        if (g_overlay_sampler) { vkDestroySampler(g_device, g_overlay_sampler, nullptr); g_overlay_sampler = VK_NULL_HANDLE; }

        for (int l = 0; l < 2; l++) {
            destroy_image(g.lat[l].img);
            destroy_image(g.lat[l].img_b);
            for (int p = 0; p < 2; p++) destroy_image(g.lat[l].bc_img[p]);
        }
        destroy_image(g_dummy);
        destroy_image(g_shot_target);
        destroy_image(g_overlay_image);
        destroy_depth();   // already gone on both normal paths; a handle that is null is skipped

        // A mapped allocation is unmapped before it is freed, which is also what makes the pointers below dangle
        // nowhere: nothing reads them after this function.
        if (g_overlay_staging_memory) { vkUnmapMemory(g_device, g_overlay_staging_memory); g_overlay_mapped = nullptr; }
        if (g_overlay_staging) { vkDestroyBuffer(g_device, g_overlay_staging, nullptr); g_overlay_staging = VK_NULL_HANDLE; }
        if (g_overlay_staging_memory) { vkFreeMemory(g_device, g_overlay_staging_memory, nullptr); g_overlay_staging_memory = VK_NULL_HANDLE; }
        if (g_overlay_geometry_memory) { vkUnmapMemory(g_device, g_overlay_geometry_memory); g_overlay_geometry_mapped = nullptr; }
        if (g_overlay_geometry) { vkDestroyBuffer(g_device, g_overlay_geometry, nullptr); g_overlay_geometry = VK_NULL_HANDLE; }
        if (g_overlay_geometry_memory) { vkFreeMemory(g_device, g_overlay_geometry_memory, nullptr); g_overlay_geometry_memory = VK_NULL_HANDLE; }
        if (g_cv_memory) { vkUnmapMemory(g_device, g_cv_memory); g_cv_mapped = nullptr; }
        if (g_cv_buffer) { vkDestroyBuffer(g_device, g_cv_buffer, nullptr); g_cv_buffer = VK_NULL_HANDLE; }
        if (g_cv_memory) { vkFreeMemory(g_device, g_cv_memory, nullptr); g_cv_memory = VK_NULL_HANDLE; }
        if (g_uniform_memory) { vkUnmapMemory(g_device, g_uniform_memory); g_uniform_mapped = nullptr; }
        if (g_uniform_buffer) { vkDestroyBuffer(g_device, g_uniform_buffer, nullptr); g_uniform_buffer = VK_NULL_HANDLE; }
        if (g_uniform_memory) { vkFreeMemory(g_device, g_uniform_memory, nullptr); g_uniform_memory = VK_NULL_HANDLE; }
        if (g_geometry) { vkDestroyBuffer(g_device, g_geometry, nullptr); g_geometry = VK_NULL_HANDLE; }
        if (g_geometry_memory) { vkFreeMemory(g_device, g_geometry_memory, nullptr); g_geometry_memory = VK_NULL_HANDLE; }

        // The pool's command buffers go with it, as a pool's allocations always do.
        if (g_command_pool) { vkDestroyCommandPool(g_device, g_command_pool, nullptr); g_command_pool = VK_NULL_HANDLE; g_command_buffer = VK_NULL_HANDLE; }

        vkDestroyDevice(g_device, nullptr);
        g_device = VK_NULL_HANDLE;
    }
#if ENABLE_VALIDATION
    // The messenger is an instance-level object and its destructor is an extension entry point, fetched the same way
    // its constructor was.
    if (g_messenger && g_instance) {
        auto destroy = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(g_instance,
                                                                                  "vkDestroyDebugUtilsMessengerEXT");
        if (destroy) destroy(g_instance, g_messenger, nullptr);
        g_messenger = VK_NULL_HANDLE;
    }
#endif
    if (g_surface && g_instance) { vkDestroySurfaceKHR(g_instance, g_surface, nullptr); g_surface = VK_NULL_HANDLE; }
    if (g_instance) { vkDestroyInstance(g_instance, nullptr); g_instance = VK_NULL_HANDLE; }
}

// ---------------------------------------------------------------------------
// The command line. The spellings are the Direct3D viewer's, and so are the two parsers: a numeric flag's value must be
// the whole argument and finite, because `--device nope` read through atoi is device 0, which is the kind of silent
// acceptance neither program allows.
// ---------------------------------------------------------------------------
static void viewer_usage(const char* exe) {
    printf("Usage: %s <PREFIX_nntc.json> [flags]\n", exe);
    printf("  The asset's descriptor and the .dds files it names, as written by nntc_encode, drawn through a Vulkan\n");
    printf("  sampler. The path is taken as given: the viewer derives no name of its own.\n\n");
    printf("  --shot FILE.bmp  render ONE frame to a 24-bit .bmp and exit, with NO window and no surface at all, so\n");
    printf("                   the frame is a function of the command line alone and runs with no desktop\n");
    printf("  --tex N          show output texture N of a material (the key N cycles them)\n");
    printf("  --cube           the cube rather than the quad (the key C)\n");
    printf("  --z F            the camera distance\n");
    printf("  --yaw F          the camera yaw, in degrees\n");
    printf("  --pitch F        the camera pitch, in degrees\n");
    printf("  --nomips         the sampler's maxLod is 0, so every fetch reads mip 0 (the key M off)\n");
    printf("  --nobias         level 1 without its LOD shift of log2(block) (the key L off), i.e. its UV gradients\n");
    printf("                   unscaled. The shift is what the encoder's 1:1 rule becomes on the GPU, so this is the\n");
    printf("                   control and not a preference\n");
    printf("  --noaniso        anisotropy off, which applies in the trilinear mode only (the key X off)\n");
    printf("  --renorm         renormalise the decoded triple as a tangent-space normal (the key V)\n");
    printf("  --raw0           show level 0's own channels instead of the decode (the key 1)\n");
    printf("  --raw1           show level 1's own channels instead of the decode (the key 2)\n");
    printf("  --bc             start with level 0 on its BC4 / BC5 pack, made at load when the file itself is\n");
    printf("                   uncompressed (the key 4 switches; lossless for 1-3 bits, an endpoint search for 4+)\n");
    printf("  --nooverlay      draw the scene without the debug strip, so two --shot frames can be compared byte for\n");
    printf("                   byte (the strip names the file and its format, which two identical renders differ in)\n");
    printf("  --size W H       the --shot frame's size, 2560x1440 by default, and the window's, 1280x720 by\n");
    printf("                   default (drag it to change it; the picture follows). The frame default exists for the\n");
    printf("                   comparison against the Direct3D viewer, whose --shot is its WINDOW's client area and\n");
    printf("                   is therefore the desktop's work area wherever that is smaller; two frames of different\n");
    printf("                   shapes are two different projections and cannot be compared\n");
    printf("  --device N       draw on physical device N instead of the first discrete GPU that can present (the\n");
    printf("                   chosen device's name is printed either way)\n");
    printf("  --coopvec 0|1    the decode through VK_NV_cooperative_vector (1) or through the plain GLSL matrix\n");
    printf("                   multiply (0); the key K switches. It is ON BY DEFAULT WHERE THE DEVICE OFFERS IT,\n");
    printf("                   and --coopvec 1 on a device that does not is refused - a run that quietly fell\n");
    printf("                   back would be a measurement of the other path\n");
    printf("  --bench N        render N frames headless, with no window and no file written, and print the mean\n");
    printf("                   and the minimum GPU time of the scene draw in microseconds. It times the CURRENT\n");
    printf("                   decode path, so the comparison is two runs that differ in --coopvec alone\n");
    printf("  --help, -h       this text\n\n");
    printf("  The shaders are view.vert and view.frag, read from beside the executable (and from nowhere else) and\n");
    printf("  compiled at runtime; the key R re-reads them from there. On a device with cooperative vectors\n");
    printf("  view_coopvec.frag is read from beside it too, and the key K switches between the two decode paths.\n");
}

static bool parse_integer(const char* text, int& out) {
    char* end = nullptr; errno = 0; const long v = strtol(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE || v < -1000000 || v > 1000000) return false;
    out = (int)v; return true;
}

static bool parse_number(const char* text, double& out) {
    char* end = nullptr; errno = 0; const double v = strtod(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE || !std::isfinite(v)) return false;
    out = v; return true;
}

int main(int argc, char** argv) {
#ifndef NDEBUG
    printf("DEBUG build\n");   // the first line out, before any argument is read, so the build type is never in doubt
#endif
    std::string json_path, shot_path;
    int device_index = -1;   // -1 = the standard choice; --device N overrides it
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") { viewer_usage(argv[0]); return 0; }   // asking for the usage is not a refusal
        else if (a == "--shot") {
            if (i + 1 >= argc) { fprintf(stderr, "ERROR: --shot needs a file name\n"); return 1; }
            shot_path = argv[++i]; g_shot = true;
        } else if (a == "--cube") g.cube = true;
        else if (a == "--nomips") g.mips_on = false;         // the key M state for --shot
        else if (a == "--nobias") g.lod_bias = false;        // the key L state
        else if (a == "--noaniso") g.aniso = false;          // the key X state
        else if (a == "--renorm") g.cb.scene.const0[3] = 1.0f;   // the key V state
        else if (a == "--raw0") g.cb.scene.const0[0] = 1.0f;     // the key 1 state
        else if (a == "--raw1") g.cb.scene.const0[1] = 1.0f;     // the key 2 state
        else if (a == "--bc") g_bc = true;                   // the key 4 state, on an asset whose level 0 is uncompressed
        else if (a == "--nooverlay") g_overlay = false;
        else if (a == "--z" || a == "--yaw" || a == "--pitch") {
            double v;
            if (i + 1 >= argc || !parse_number(argv[i + 1], v)) {
                fprintf(stderr, "ERROR: %s needs a number, not '%s'\n", a.c_str(), i + 1 < argc ? argv[i + 1] : "");
                return 1;
            }
            i++;
            if (a == "--z") g.z = (float)v; else if (a == "--yaw") g.yaw = (float)v; else g.pitch = (float)v;
        } else if (a == "--tex") {
            int v;
            if (i + 1 >= argc || !parse_integer(argv[i + 1], v)) {
                fprintf(stderr, "ERROR: --tex needs a texture index, not '%s'\n", i + 1 < argc ? argv[i + 1] : "");
                return 1;
            }
            i++; g.tex_shown = v;
        } else if (a == "--size") {
            int w = 0, h = 0;
            if (i + 2 >= argc || !parse_integer(argv[i + 1], w) || !parse_integer(argv[i + 2], h) ||
                w < 16 || h < 16 || w > 16384 || h > 16384) {
                fprintf(stderr, "ERROR: --size needs a width and a height in 16..16384, not '%s %s'\n",
                        i + 1 < argc ? argv[i + 1] : "", i + 2 < argc ? argv[i + 2] : "");
                return 1;
            }
            i += 2; FRAME_WIDTH = w; FRAME_HEIGHT = h; WINDOW_WIDTH = w; WINDOW_HEIGHT = h;
        } else if (a == "--coopvec") {
            int v;
            if (i + 1 >= argc || !parse_integer(argv[i + 1], v) || (v != 0 && v != 1)) {
                fprintf(stderr, "ERROR: --coopvec needs 0 or 1, not '%s'\n", i + 1 < argc ? argv[i + 1] : "");
                return 1;
            }
            i++; g_cv_want = v;
        } else if (a == "--bench") {
            int v;
            if (i + 1 >= argc || !parse_integer(argv[i + 1], v) || v < 1 || v > 100000) {
                fprintf(stderr, "ERROR: --bench needs a frame count in 1..100000, not '%s'\n",
                        i + 1 < argc ? argv[i + 1] : "");
                return 1;
            }
            i++; g_bench_frames = v;
        } else if (a == "--device") {
            if (i + 1 >= argc || !parse_integer(argv[i + 1], device_index) || device_index < 0) {
                fprintf(stderr, "ERROR: --device needs a device index, not '%s'\n", i + 1 < argc ? argv[i + 1] : "");
                return 1;
            }
            i++;
        } else if (!a.empty() && a[0] == '-') {
            fprintf(stderr, "ERROR: unknown option '%s'\n", a.c_str());
            viewer_usage(argv[0]);
            return 1;
        } else if (json_path.empty()) json_path = a;
        else { fprintf(stderr, "ERROR: one descriptor only; '%s' is a second\n", a.c_str()); return 1; }
    }
    // Nothing is printed about the run's state before it is known that there IS a run: named no asset, the program
    // owes the caller the usage and nothing else.
    if (json_path.empty()) {
        viewer_usage(argv[0]);
        return 1;   // no asset was named, which IS a refusal, unlike --help above
    }
    // Both are the headless path and each owns what its run prints. Asking for both would write a file AND a
    // timing and leave a reader to guess which of the N frames the file is, so it is a refusal.
    if (g_shot && g_bench_frames > 0) {
        fprintf(stderr, "ERROR: --shot writes one frame and --bench times N; ask for one of them\n");
        return 1;
    }
    // The descriptor is opened here, before an instance, a device or a shader exists, so that a path this program
    // cannot read costs one line and an exit rather than a window that appears and vanishes. "cannot read" would cover
    // three different mistakes, because an empty read is all three, so the caller is told which: a path that is not
    // there, a directory named where a descriptor was meant, and a file that exists but holds nothing.
    {
        std::error_code ec;
        const std::filesystem::path p(json_path);
        if (!std::filesystem::exists(p, ec)) { fprintf(stderr, "ERROR: cannot read '%s'\n", json_path.c_str()); return 1; }
        if (!std::filesystem::is_regular_file(p, ec)) { fprintf(stderr, "ERROR: '%s' is not a file\n", json_path.c_str()); return 1; }
        if (read_file(json_path).empty()) { fprintf(stderr, "ERROR: '%s' is empty\n", json_path.c_str()); return 1; }
    }
    // `level 0` here is what the VIEWER will bind, which for a file that is already block-compressed is the file's own
    // blocks and not a pack of this program's making; load_asset says which, once it has read the file.
    // --shot's destination, checked before an instance, a device or a single upload exists. A directory that is not
    // there is a typo, and a typo is worth one line now rather than after the whole asset has been read and drawn; the
    // write itself keeps its own checks, because a directory can go away between this line and that one.
    if (g_shot) {
        const std::filesystem::path out(shot_path);
        const std::filesystem::path dir = out.has_parent_path() ? out.parent_path() : std::filesystem::path(".");
        std::error_code ec;
        if (!std::filesystem::is_directory(dir, ec)) {
            fprintf(stderr, "ERROR: cannot write '%s': '%s' is not a directory this program can see\n",
                    shot_path.c_str(), dir.string().c_str());
            return 1;
        }
    }
    printf("start: mips %s, level 1 LOD shift %s, anisotropy %s, level 0 %s\n", g.mips_on ? "on" : "off",
           g.lod_bias ? "on" : "off", g.aniso ? "on (trilinear only)" : "off",
           g_bc ? "the BC pack, if the file is uncompressed" : "as the file holds it");
    const int status = (g_shot || g_bench_frames > 0)
                           ? render_shot(shot_path, json_path, device_index, argv[0], g_bench_frames)
                           : render_window(json_path, device_index, argv[0]);
    destroy_everything();
    return status;
}
