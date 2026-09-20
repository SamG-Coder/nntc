// nntc_view_d3d12: the same asset, on Direct3D 12.
//
// A third viewer for what nntc_encode writes (docs/FORMAT.md), beside the Direct3D 11 one in viewer/ and the Vulkan one in
// viewer_vk/. The design and the reasoning behind every choice in it are docs/D3D12_LINEAR_ALGEBRA_PLAN.md, part I; this file
// is that plan's phase A. Derived from viewer/main.cpp (Apache 2.0, LICENSE at the root of this tree): the asset load, the
// camera, the quad and the cube, the overlay, the keys and the flags are that program's, transliterated, and what is new here
// is the Direct3D 12 plumbing around them - heaps, a root signature, barriers, upload and readback copies, a fence and two
// pipeline state objects.
//
// THE SHADER IS THE DIRECT3D 11 VIEWER'S OWN, unchanged: viewer/bin/nntc_view.hlsl, compiled by the same D3DCompileFromFile to
// the same vs_5_0 / ps_5_0 DXBC. Direct3D 12 accepts DXBC, so the two viewers hand their drivers the same bytes and any
// difference between their frames is in the api plumbing, which is the thing this viewer exists to check (the plan's section
// A.3). Nothing here re-reads the decode, the sampling or the feature order.
//
// Usage: nntc_view_d3d12 <PREFIX_nntc.json> [flags]
//        --shot FILE.bmp renders ONE frame and exits, with NO WINDOW AT ALL, and takes no keyboard input, so its bytes are a
//        function of the command line and the asset and of nothing else.
// Controls: the Direct3D 11 viewer's keys, letter for letter (F1 lists them).
//
// Dependencies: the C runtime, Win32, d3d12, dxgi and d3dcompiler, all from the Windows SDK. No Agility SDK, no preview
// package, no Developer Mode: the viewer runs on any Direct3D 12 device at feature level 11_0.
//
// PHASE B, and it is OPTIONAL IN EVERY SENSE. A build made with the CMake option NNTC_D3D12_LINALG (off by default) also
// carries a second pixel shader, viewer_d3d12/bin/nntc_view_linalg.hlsl, whose decode is one Shader Model 6.10
// MultiplyAdd instead of two loops over W - the Direct3D 12 counterpart of the Vulkan viewer's cooperative-vector path.
// That build depends on three preview NuGet packages and on Developer Mode, and it still draws the plain path on every
// device whose query does not pass. WITHOUT the option nothing below changes: no preview header is included, no
// D3D12SDKVersion is exported, no second shader is compiled and the frames are the same bytes. Everything that is
// specific to it is between #if NNTC_D3D12_LINALG and its #endif, and the design is docs/D3D12_LINEAR_ALGEBRA_PLAN.md
// part II. viewer_d3d12/PHASE_B_NOTES.md is the diagnosis guide for a machine where it does not work.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>

#include <cstdio>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>

#if NNTC_D3D12_LINALG
// <d3d12.h> above is the AGILITY SDK's, not the Windows SDK's: the build puts the preview package's include directory
// first on the path, which is what brings D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT, D3D_SHADER_MODEL_6_10 and the two
// Preview interfaces into scope. dxcapi.h is the preview DXC's, and dxcompiler.dll and dxil.dll travel beside the
// executable with it.
#include <dxcapi.h>
#include <combaseapi.h>
#endif

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

#if NNTC_D3D12_LINALG
#pragma comment(lib, "version.lib")   // GetFileVersionInfo, for the D3D12Core.dll line of the diagnostics

// The two symbols the redistributable specification defines, which is how the loader finds the preview runtime in the
// D3D12\ folder beside this executable rather than the system one (https://microsoft.github.io/DirectX-Specs/d3d/D3D12Redistributable.html).
// They are the reason the path cannot be a run-time switch alone and the option has to be a compile-time one: a retail
// build must not carry them.
extern "C" {
__declspec(dllexport) extern const UINT D3D12SDKVersion = D3D12_PREVIEW_SDK_VERSION;
__declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";
}
#endif

#ifdef _DEBUG
#define ENABLE_D3D_DEBUG 1
#else
#define ENABLE_D3D_DEBUG 0
#endif

template <typename T> static void safe_release(T*& p) { if (p) { p->Release(); p = nullptr; } }

// ---------------------------------------------------------------------------
// 8x8 debug font (g_debug_font8x8_basic from encoder/basisu_enc.cpp, ASCII 32-127).
// Bit order: pixel (x,y) set if (glyph[y] >> x) & 1  (LSB = leftmost, y=0 = top).
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
//
// Two sizes, as the Vulkan viewer has them and for the same reason. The FRAME is what --shot writes and is 2560x1440 by
// default, which is the Direct3D 11 viewer's window and therefore the shape its own --shot produces; a comparison of two
// frames of different shapes is a comparison of two different projections. The WINDOW opens at 1280x720, because a
// 2560-wide window does not fit every desktop, and it is resizable. --size sets both.
// ---------------------------------------------------------------------------
static int   FRAME_WIDTH  = 2560, FRAME_HEIGHT = 1440;
static int   WINDOW_WIDTH = 1280, WINDOW_HEIGHT = 720;
static int   g_frame_w = 0, g_frame_h = 0;   // what the frame being recorded right now is: the window's client area, or the shot's
static const float FOV_DEGREES = 90.0f;
static const float Z_MIN = 0.40f, Z_MAX = -50.0f;
static const float Z_SPEED = 1.0f, XY_SPEED = 0.75f, ROT_SPEED = 90.0f;

// The sampler set. [mips on / off][0=point, 1=bilinear, 2=trilinear, 3=anisotropic]: the MaxLOD = 0 row is key M (the
// hardware held at mip 0 of both textures, nothing reloaded), and slot 3 is the trilinear filter with anisotropy, which
// key X turns on and off. There is ONE set, shared by both levels: level 1's LOD shift is a scale on its gradients and
// not a sampler field, so it composes with every filter and with the mips toggle for free.
static const int FILTER_STATES = 4;
static const UINT ANISO_MAX = 8;   // the anisotropy asked for when it is on; one line to change

// The descriptor heaps' fixed layout. A shader-visible CBV/SRV/UAV heap of nine and a sampler heap of sixteen, both written
// once and only indexed after that, because nothing in this viewer's frame needs a descriptor it did not know about at load.
//
// The samplers are stored in PAIRS - each of the eight states written twice, at 2k and 2k+1 - so that one descriptor table of
// two samplers gives s0 and s1 the SAME state, which is what the Direct3D 11 viewer binds (both levels through one sampler;
// level 1's LOD shift is const0.z and not a sampler field). The alternative, two root tables into one heap, costs a root
// parameter and says less.
//
// A build with the linear-algebra option has FOUR descriptors per set rather than three: the fourth is the raw buffer at
// t3 that the Shader Model 6.10 pixel shader reads its matrix and bias from. The sets are written out of these constants
// and never out of literals, so a build without the option has the three-descriptor layout it always had, to the slot.
#if NNTC_D3D12_LINALG
static const int SRV_PER_SET = 4;
#else
static const int SRV_PER_SET = 3;
#endif
static const int SRV_SET_PLAIN  = 0;                 // t0, t1, t2 with level 0 as the file holds it (and t3, the weights, under the option)
static const int SRV_SET_PACKED = SRV_PER_SET;       // the same with level 0 from the load-time BC4 / BC5 pack (key 4)
static const int SRV_OVERLAY    = 2 * SRV_PER_SET;   // the overlay texture at t0, with null SRVs behind it so every descriptor of the table is valid
static const int SRV_HEAP_COUNT = 3 * SRV_PER_SET;
static const int SAMPLER_HEAP_COUNT = 2 * 2 * FILTER_STATES;

// ---------------------------------------------------------------------------
// The Direct3D 12 objects. One direct queue, one command list, two allocators and a fence: the frame loop waits for the GPU
// at the end of every frame, which is the simplest correct loop and is what this viewer wants - it is a sample and a test,
// and speed is not one of its goals (the plan's section A.5).
// ---------------------------------------------------------------------------
static IDXGIFactory4*            g_factory = nullptr;
static IDXGIAdapter1*            g_adapter = nullptr;
static ID3D12Device*             g_dev = nullptr;
static ID3D12CommandQueue*       g_queue = nullptr;
static ID3D12CommandAllocator*   g_alloc[2] = { nullptr, nullptr };
static ID3D12GraphicsCommandList* g_list = nullptr;
static ID3D12Fence*              g_fence = nullptr;
static UINT64                    g_fence_value = 0;
static HANDLE                    g_fence_event = nullptr;
static IDXGISwapChain3*          g_swapchain = nullptr;
static ID3D12Resource*           g_backbuffer[2] = { nullptr, nullptr };
static ID3D12Resource*           g_shot_target = nullptr;   // the offscreen render target of the headless --shot
static ID3D12Resource*           g_depth = nullptr;
static ID3D12DescriptorHeap*     g_rtv_heap = nullptr;      // 3: the two back buffers and the offscreen target
static ID3D12DescriptorHeap*     g_dsv_heap = nullptr;      // 1
static ID3D12DescriptorHeap*     g_srv_heap = nullptr;      // SRV_HEAP_COUNT, shader visible
static ID3D12DescriptorHeap*     g_samp_heap = nullptr;     // SAMPLER_HEAP_COUNT, shader visible
static ID3D12RootSignature*      g_rootsig = nullptr;
static ID3D12Resource*           g_cb_scene = nullptr, * g_cb_dec = nullptr;   // an upload heap, mapped once and written per frame
static uint8_t*                  g_cb_scene_ptr = nullptr, * g_cb_dec_ptr = nullptr;
static UINT                      g_rtv_step = 0, g_dsv_step = 0, g_srv_step = 0, g_samp_step = 0;
static int                       g_frame_index = 0;   // which of the two command allocators this frame records into
static bool                      g_quit = false;
static bool                      g_bc = false;   // the level-0 textures bound: false = the .dds as it came (the default), true = the BC4 / BC5 pack made at load (key 4; --bc starts packed)

// The load-time command list's work: every texture upload is recorded on the frame list before the first frame and flushed
// once, and the staging buffers are held alive until that flush has completed on the GPU.
static std::vector<ID3D12Resource*> g_upload_keepalive;

// The decode path. Phase A has one, and the plan's phase B adds a second (a Shader Model 6.10 linear-algebra pixel shader,
// docs/D3D12_LINEAR_ALGEBRA_PLAN.md part II) behind a CMake option that is off by default and a run-time query. Everything
// that would have to know about a second path goes through this enum, the two pipeline state objects built in
// load_shader, and decode_path_name below, so that adding it is a local change rather than a reading of the frame loop.
enum DecodePath { DECODE_PLAIN = 0, DECODE_LINALG = 1 };
static DecodePath g_decode = DECODE_PLAIN;
static const char* decode_path_name() { return g_decode == DECODE_PLAIN ? "plain" : "linear algebra"; }

// The linear-algebra path's own state, and the three flags that steer it. g_la_want is --linalg: -1 not given, so the
// path is taken wherever the query passes; 0 the plain path, always honoured, on every build; 1 a REFUSAL where the
// query did not pass, because a run that quietly fell back would be a measurement of the other path under this one's
// name. The reason a query did not pass is kept, not just the fact, because it is what every refusal and every skip
// prints (the plan's B.3).
static bool        g_la_available = false;   // the device query passed, so the second pipeline can exist
static std::string g_la_why_not;             // and when it did not, which step of it failed
static int         g_la_want = -1;           // --linalg
static bool        g_la_fp32 = false;        // the row that was granted keeps the bias and the result in fp32
static bool        g_la_mul_optimal = false; // the matrix goes through ConvertLinearAlgebraMatrix into the device's own layout
static int         g_la_layout_want = -1;    // --linalg-layout: -1 the device's preference, 0 row-major, 1 multiply-optimal
static int         g_bench_frames = 0;       // --bench N: N headless frames timed, no file written

// The padded shape. M and K are template arguments of the shader's Matrix and therefore compile-time constants, so the
// matrix is padded to the largest shape this format allows: nout <= 18 rows of nin <= 24 columns.
static const UINT LA_M = 18, LA_K = 24;

static const float LOD_BIAS_LEVEL1_MAX = 8.0f;   // a block of 256: far past anything the format writes, and still a finite scale
static bool check_level1_lod_shift(int block, float bias) {
    if (!(bias >= 0.0f && bias <= LOD_BIAS_LEVEL1_MAX)) {
        fprintf(stderr, "ERROR: the descriptor's lod_bias_level1 is %g, which is outside [0, %g]: it is the number of mip levels\n"
                        "       level 1 is shifted by, so it cannot be negative and cannot be that large\n", (double)bias, (double)LOD_BIAS_LEVEL1_MAX);
        return false;
    }
    printf("  level 1 LOD: UV gradients scaled by %g (2^lod_bias_level1, lod_bias_level1 = %g%s); key L toggles it\n",
           (double)std::exp2(bias), (double)bias, bias == std::log2((float)block) ? ", log2 of the block" : ", NOT log2 of the block");
    return true;
}

// The asset: one latent texture per level. The Direct3D 11 viewer's LatentTex with its views replaced by resources - the
// descriptors themselves live in the one shader-visible heap and are written by refresh_srv_sets.
struct LatentTex {
    ID3D12Resource* tex = nullptr;
    ID3D12Resource* bc_tex[2] = { nullptr, nullptr };   // level 0 packed: channels 0-1 (BC5, or BC4 for one channel) and what is left (a fourth channel makes a BC5, a third alone a BC4), made at load
    int bc_n = 0; int bc_nc[2] = { 0, 0 }; double bc_psnr = 0, bc_maxerr = 0; bool bc_lossless = false;   // the packed texture count, each one's channel count, the packing quality over the chain, the round trip's largest error
    std::vector<std::vector<uint8_t>> raw; std::vector<std::pair<int, int>> dims;   // the .dds levels as read (for the pack)
    ID3D12Resource* tex_b = nullptr;   // the SECOND file of a two-file level 0 (channels 2-3 as a BC5, or channel 2 alone as a BC4), bound to t2
    int file_files = 1;    // the .dds files this level is stored in: one, or two for a BC level 0 of three or four channels
    bool file_bc = false;  // the level came block-compressed in the file, so the viewer makes no pack of its own
    int W = 0, H = 0, C = 0, stored_C = 0, mips = 0, dxgi = 0;
    // The SECOND file's own format and channel count, which are not the first's: a three-channel level 0 is a BC5 of
    // channels 0-1 and a BC4 of channel 2 alone, so the two files differ in both. Zero when there is only one file.
    int stored_C_b = 0, dxgi_b = 0;
    float lo[4] = {0,0,0,0}, hi[4] = {1,1,1,1};
    int bits[4] = {8,8,8,8};
    std::string file;
};
// The decoder constant buffer: must match cbuffer DecoderConstants in nntc_view.hlsl (16-byte rules).
struct DecCB {
    float lo0[4], hi0[4], lo1[4], hi1[4];
    int   dims[4];        // C0, C1, nin, nout
    int   sel[4];         // terms mask, output texture shown, 0, 0
    float W[108][4];      // row r, column c at W[r * 6 + c / 4][c % 4]: six float4 per row, MAX_TEXTURES * 3 rows
    float bias[5][4];     // nout <= 18, so five float4 cover it
#if NNTC_D3D12_LINALG
    // Where the linear-algebra shader's bias sits in the weights buffer and what the matrix's stride is. Both follow the
    // layout the run actually settled on, so they arrive as constants rather than as numbers baked into the shader: under
    // the multiply-optimal layout the stride is the 0 the application supplies, because that layout's addressing is the
    // device's own business and only its SIZE comes back from the device, and under row-major it is the 48 bytes of one
    // row this program writes itself. They are at the END of the struct, so a build without the option writes exactly the
    // bytes it always wrote.
    uint32_t linalg[4];   // x = the bias's byte offset, y = the matrix's stride in bytes, z = 0, w = 0
#endif
};

struct State {
    float x = 0, y = 0, z = -3.0f, yaw = 0, pitch = 0;
    bool  cube = false;
    int   filter_mode = 2;
    bool  aniso = true;      // key X: anisotropic filtering, which applies in TRILINEAR mode only
    bool  lod_bias = true;   // key L: level 1's UV gradients scaled by 2^lod_bias_level1 (the encoder's 1:1 rule)
    float lod_bias_level1 = 2.0f;   // the descriptor's own value: the mip levels level 1 is shifted by, so the gradient scale is 2^this
    bool  mips_on = true;    // key M: false = the sampler's MaxLOD is 0, so every fetch reads mip 0 (the textures unchanged)
    ID3D12PipelineState* pso_scene = nullptr;
    ID3D12PipelineState* pso_overlay = nullptr;
#if NNTC_D3D12_LINALG
    ID3D12PipelineState* pso_linalg = nullptr;   // the Shader Model 6.10 scene pipeline, where the query passed
#endif
    LatentTex lat[2];
    DecCB  dec{};
    int    textures_out = 1, tex_shown = 0;    // a material decodes 3 * textures_out channels; the shader shows one triple
    int    src_w = 0, src_h = 0, block = 4;
    std::string terms;
    float  const0[4] = {0,0,0,0};
    float  const1[4] = {0,0,0,0};
    ID3D12Resource* debug_tex = nullptr;       // the overlay strip, uploaded when the text changes
    ID3D12Resource* debug_upload = nullptr;
    ID3D12Resource* debug_vb = nullptr;        // four vertices in an upload heap, rewritten per frame
    ID3D12Resource* debug_ib = nullptr;
    bool   debug_dirty = true;
    bool   debug_in_ps = false;   // the overlay texture's resource state: false until its first upload has been recorded
    std::string shader_path;
#if NNTC_D3D12_LINALG
    std::string linalg_shader_path;   // nntc_view_linalg.hlsl, beside the executable as the plain one is
#endif
};
static State g;

static const char* filter_name(int m) { return m == 0 ? "POINT" : (m == 2 ? "TRILINEAR" : "BILINEAR"); }

// The sampler slot a filter state picks: anisotropy is the trilinear filter with several samples along the footprint's
// long axis, so it is a fourth state of the trilinear mode and not a mode of its own.
static int filter_slot() { return (g.filter_mode == 2 && g.aniso) ? 3 : g.filter_mode; }

// ---------------------------------------------------------------------------
// 4x4 matrices (row-major with column-vector convention; one transpose at cbuffer-write time).
// ---------------------------------------------------------------------------
struct Mat4 { float m[16]; };
static Mat4 mat_identity() { Mat4 r{}; for (int i=0;i<4;i++) r.m[i*4+i]=1.0f; return r; }
static Mat4 mat_mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int i=0;i<4;i++) for (int j=0;j<4;j++) { float s=0; for(int k=0;k<4;k++) s+=a.m[i*4+k]*b.m[k*4+j]; r.m[i*4+j]=s; }
    return r;
}
static Mat4 mat_perspective(float fov_deg, float aspect, float znear, float zfar) {
    Mat4 m{};
    float f = 1.0f / std::tan((fov_deg * 3.14159265358979f / 180.0f) / 2.0f);
    m.m[0*4+0] = f / aspect;
    m.m[1*4+1] = f;
    m.m[2*4+2] = zfar / (znear - zfar);
    m.m[2*4+3] = (zfar * znear) / (znear - zfar);
    m.m[3*4+2] = -1.0f;
    return m;
}
static Mat4 mat_translate(float x, float y, float z) { Mat4 m = mat_identity(); m.m[0*4+3]=x; m.m[1*4+3]=y; m.m[2*4+3]=z; return m; }
static Mat4 mat_rot_y(float deg) {
    Mat4 m = mat_identity(); float r=deg*3.14159265358979f/180.0f, c=std::cos(r), s=std::sin(r);
    m.m[0*4+0]=c; m.m[0*4+2]=s; m.m[2*4+0]=-s; m.m[2*4+2]=c; return m;
}
static Mat4 mat_rot_x(float deg) {
    Mat4 m = mat_identity(); float r=deg*3.14159265358979f/180.0f, c=std::cos(r), s=std::sin(r);
    m.m[1*4+1]=c; m.m[1*4+2]=-s; m.m[2*4+1]=s; m.m[2*4+2]=c; return m;
}
static Mat4 mat_transpose(const Mat4& a) { Mat4 r{}; for (int i=0;i<4;i++) for (int j=0;j<4;j++) r.m[i*4+j]=a.m[j*4+i]; return r; }

// ---------------------------------------------------------------------------
// Small Direct3D 12 helpers: the descriptor arithmetic and the two resource shapes this viewer creates.
// ---------------------------------------------------------------------------
static D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle(ID3D12DescriptorHeap* heap, UINT step, int index) {
    D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += (SIZE_T)step * (SIZE_T)index;
    return h;
}
static D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle(ID3D12DescriptorHeap* heap, UINT step, int index) {
    D3D12_GPU_DESCRIPTOR_HANDLE h = heap->GetGPUDescriptorHandleForHeapStart();
    h.ptr += (UINT64)step * (UINT64)index;
    return h;
}
static D3D12_HEAP_PROPERTIES heap_props(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = type; hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN; hp.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    hp.CreationNodeMask = 1; hp.VisibleNodeMask = 1;
    return hp;
}
static D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = bytes; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = D3D12_RESOURCE_FLAG_NONE;
    return rd;
}
static D3D12_RESOURCE_DESC texture_desc(DXGI_FORMAT fmt, int W, int H, int mips) {
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; rd.Width = (UINT64)W; rd.Height = (UINT)H; rd.DepthOrArraySize = 1;
    rd.MipLevels = (UINT16)mips; rd.Format = fmt; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = D3D12_RESOURCE_FLAG_NONE;
    return rd;
}
static void barrier(ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    b.Transition.pResource = r; b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from; b.Transition.StateAfter = to;
    g_list->ResourceBarrier(1, &b);
}
static void wait_for_gpu() {
    const UINT64 want = ++g_fence_value;
    if (FAILED(g_queue->Signal(g_fence, want))) return;
    if (g_fence->GetCompletedValue() < want) {
        if (SUCCEEDED(g_fence->SetEventOnCompletion(want, g_fence_event))) WaitForSingleObject(g_fence_event, INFINITE);
    }
}
static bool begin_commands() {
    if (FAILED(g_alloc[g_frame_index]->Reset())) return false;
    return SUCCEEDED(g_list->Reset(g_alloc[g_frame_index], nullptr));
}
static bool end_commands_and_wait() {
    if (FAILED(g_list->Close())) return false;
    ID3D12CommandList* lists[] = { g_list };
    g_queue->ExecuteCommandLists(1, lists);
    wait_for_gpu();
    g_frame_index = 1 - g_frame_index;
    return true;
}

// One committed texture in the default heap with its whole mip chain uploaded through a staging buffer. The row pitch a
// copy wants is the GPU's and not the file's - Direct3D 12 asks for 256-byte-aligned rows and the .dds is packed tight, and
// for a block format a "row" is a row of 4x4 BLOCKS - so the footprints come from GetCopyableFootprints rather than from
// arithmetic here. The staging buffer is kept alive until the load flush, because the copy is a GPU command and not a memcpy.
static ID3D12Resource* create_texture_with_levels(DXGI_FORMAT fmt, int W, int H, int mips,
                                                  const std::vector<const uint8_t*>& level_bytes,
                                                  const std::vector<size_t>& level_pitch, const char* what) {
    const D3D12_RESOURCE_DESC rd = texture_desc(fmt, W, H, mips);
    const D3D12_HEAP_PROPERTIES def = heap_props(D3D12_HEAP_TYPE_DEFAULT);
    ID3D12Resource* tex = nullptr;
    HRESULT hr = g_dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&tex));
    if (FAILED(hr)) { fprintf(stderr, "ERROR: the texture for %s could not be created (0x%08X)\n", what, (unsigned)hr); return nullptr; }

    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> fp((size_t)mips);
    std::vector<UINT> rows((size_t)mips);
    std::vector<UINT64> row_bytes((size_t)mips);
    UINT64 total = 0;
    g_dev->GetCopyableFootprints(&rd, 0, (UINT)mips, 0, fp.data(), rows.data(), row_bytes.data(), &total);

    const D3D12_HEAP_PROPERTIES up = heap_props(D3D12_HEAP_TYPE_UPLOAD);
    const D3D12_RESOURCE_DESC ubd = buffer_desc(total);
    ID3D12Resource* staging = nullptr;
    hr = g_dev->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &ubd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&staging));
    if (FAILED(hr)) { fprintf(stderr, "ERROR: the upload buffer for %s could not be created (0x%08X)\n", what, (unsigned)hr); tex->Release(); return nullptr; }
    uint8_t* mapped = nullptr;
    D3D12_RANGE none{ 0, 0 };
    if (FAILED(staging->Map(0, &none, (void**)&mapped))) { fprintf(stderr, "ERROR: the upload buffer for %s could not be mapped\n", what); staging->Release(); tex->Release(); return nullptr; }
    for (int i = 0; i < mips; i++)
        for (UINT y = 0; y < rows[(size_t)i]; y++)
            memcpy(mapped + fp[(size_t)i].Offset + (SIZE_T)y * fp[(size_t)i].Footprint.RowPitch,
                   level_bytes[(size_t)i] + (size_t)y * level_pitch[(size_t)i], (size_t)row_bytes[(size_t)i]);
    staging->Unmap(0, nullptr);

    for (int i = 0; i < mips; i++) {
        D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource = tex; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dst.SubresourceIndex = (UINT)i;
        D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource = staging; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = fp[(size_t)i];
        g_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    barrier(tex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    g_upload_keepalive.push_back(staging);
    return tex;
}

// ---------------------------------------------------------------------------
// Shader loading (viewer/bin/nntc_view.hlsl, copied beside the executable by the build and compiled at runtime: edit it
// there, build, and press R).
//
// It is read FROM BESIDE THE EXECUTABLE AND FROM NOWHERE ELSE, which is the Vulkan viewer's rule rather than the Direct3D 11
// viewer's (that one looks in the working directory first): a stray nntc_view.hlsl in the directory a run was started from
// silently replacing the real one is a mistake nobody enjoys finding, and the path is printed on every compile.
// ---------------------------------------------------------------------------
static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::string();
    std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}
// A path from argv is in the process's ANSI code page, so widening it one char at a time turns every byte above 127
// into the wrong character (and a two-byte character into two of them). D3DCompileFromFile wants wide, so it is
// converted properly.
static std::wstring widen(const std::string& s) {
    if (s.empty()) return std::wstring();
    const int n = MultiByteToWideChar(CP_ACP, 0, s.c_str(), (int)s.size(), nullptr, 0);
    if (n <= 0) return std::wstring(s.begin(), s.end());
    std::wstring w((size_t)n, L'\0');
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}
static std::string narrow(const wchar_t* s) {
    const int n = WideCharToMultiByte(CP_ACP, 0, s, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return std::string();
    std::string a((size_t)(n - 1), '\0');
    WideCharToMultiByte(CP_ACP, 0, s, -1, &a[0], n, nullptr, nullptr);
    return a;
}
static std::string exe_dir() {
    char buf[MAX_PATH * 2] = { 0 };
    const DWORD n = GetModuleFileNameA(nullptr, buf, (DWORD)sizeof(buf));
    if (!n || n >= sizeof(buf)) return std::string();
    std::string a(buf, buf + n);
    const size_t slash = a.find_last_of("/\\");
    return slash == std::string::npos ? std::string() : a.substr(0, slash + 1);
}
static ID3DBlob* compile_hlsl_file(const std::string& path, const char* entry, const char* target) {
    const std::wstring wpath = widen(path);
    ID3DBlob* code = nullptr, * errors = nullptr;
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
    HRESULT hr = D3DCompileFromFile(wpath.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry, target, flags, 0, &code, &errors);
    if (errors) { fprintf(stderr, "%s (%s):\n%s\n", FAILED(hr) ? "SHADER ERROR" : "SHADER WARNING", entry, (const char*)errors->GetBufferPointer()); errors->Release(); }
    if (FAILED(hr)) {
        if (!code) fprintf(stderr, "SHADER ERROR (%s): '%s' could not be compiled (0x%08X)\n", entry, path.c_str(), (unsigned)hr);
        safe_release(code); return nullptr;
    }
    return code;
}

#if NNTC_D3D12_LINALG
// The SECOND compiler. nntc_view_linalg.hlsl is Shader Model 6.10 and fxc stops at 5.1, so this path's two stages are
// DXIL from the preview DXC beside the executable. -enable-16bit-types is what makes `half` a real fp16 rather than a
// minimum-precision float, and -I names the directory dx/linalg.h was copied into, which is that same directory.
//
// The include path is the EXECUTABLE'S directory and not the working directory, for the reason compile_hlsl_file gives
// about the shader itself: a stray dx/linalg.h where a run happened to start is not the one this build was tested with.
//
// dxcompiler.dll is loaded BY HAND rather than linked against its import library, so that an executable copied away from
// the files the build put beside it still RUNS and still draws the plain path. An import would make the missing DLL a
// loader error before main, which is the one failure this program could not explain.
static DxcCreateInstanceProc g_dxc_create = nullptr;
static bool load_dxc() {
    if (g_dxc_create) return true;
    const HMODULE dll = LoadLibraryExA("dxcompiler.dll", nullptr, LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!dll) return false;
    g_dxc_create = (DxcCreateInstanceProc)GetProcAddress(dll, "DxcCreateInstance");
    return g_dxc_create != nullptr;
}
static IDxcBlob* compile_dxil_file(const std::string& path, const wchar_t* entry, const wchar_t* target,
                                   bool fp32, bool mul_optimal) {
    if (!load_dxc()) { fprintf(stderr, "SHADER ERROR: dxcompiler.dll could not be loaded from beside the executable, so nothing can be compiled for Shader Model 6.10\n"); return nullptr; }
    IDxcUtils* utils = nullptr; IDxcCompiler3* compiler = nullptr;
    if (FAILED(g_dxc_create(CLSID_DxcUtils, IID_PPV_ARGS(&utils)))) { fprintf(stderr, "SHADER ERROR: IDxcUtils could not be created\n"); return nullptr; }
    if (FAILED(g_dxc_create(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler)))) { fprintf(stderr, "SHADER ERROR: IDxcCompiler3 could not be created\n"); safe_release(utils); return nullptr; }
    const std::wstring wpath = widen(path);
    IDxcBlobEncoding* source = nullptr;
    HRESULT hr = utils->LoadFile(wpath.c_str(), nullptr, &source);
    if (FAILED(hr)) { fprintf(stderr, "SHADER ERROR: '%s' could not be read (0x%08X)\n", path.c_str(), (unsigned)hr); safe_release(compiler); safe_release(utils); return nullptr; }
    DxcBuffer buffer{ source->GetBufferPointer(), source->GetBufferSize(), DXC_CP_ACP };
    const std::wstring include_dir = widen(exe_dir());
    const std::wstring fp32_def = std::wstring(L"NNTC_LA_FP32=") + (fp32 ? L"1" : L"0");
    const std::wstring opt_def = std::wstring(L"NNTC_LA_MUL_OPTIMAL=") + (mul_optimal ? L"1" : L"0");
    const wchar_t* args[] = { L"-E", entry, L"-T", target, L"-enable-16bit-types",
                              L"-I", include_dir.c_str(), L"-D", fp32_def.c_str(), L"-D", opt_def.c_str() };
    IDxcIncludeHandler* includes = nullptr;
    utils->CreateDefaultIncludeHandler(&includes);
    IDxcResult* result = nullptr;
    hr = compiler->Compile(&buffer, args, (UINT32)(sizeof(args) / sizeof(args[0])), includes, IID_PPV_ARGS(&result));
    IDxcBlob* code = nullptr;
    if (result) {
        IDxcBlobUtf8* errors = nullptr;
        result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
        HRESULT status = E_FAIL;
        result->GetStatus(&status);
        if (errors && errors->GetStringLength())
            fprintf(stderr, "%s (%ls):\n%s\n", FAILED(status) ? "SHADER ERROR" : "SHADER WARNING", entry, errors->GetStringPointer());
        safe_release(errors);
        if (SUCCEEDED(status)) result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&code), nullptr);
        else if (SUCCEEDED(hr)) hr = status;
    }
    if (!code) fprintf(stderr, "SHADER ERROR (%ls): '%s' could not be compiled for %ls (0x%08X)\n", entry, path.c_str(), target, (unsigned)hr);
    safe_release(result); safe_release(includes); safe_release(source); safe_release(compiler); safe_release(utils);
    return code;
}
#endif

// The two tiny shaders the overlay quad is drawn with, as string constants: a bitmap font is not api-specific and neither
// is a textured quad, so they are the Direct3D 11 viewer's own and are compiled to DXBC beside the scene shader.
static const char* DEBUG_HLSL =
    "struct VSO { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
    "VSO VSMain(float2 pos : POSITION, float2 uv : TEXCOORD0) {\n"
    "    VSO o; o.pos = float4(pos, 0.0, 1.0); o.uv = uv; return o; }\n"
    "Texture2D tex : register(t0); SamplerState samp : register(s0);\n"
    "float4 PSMain(VSO i) : SV_Target { return tex.Sample(samp, i.uv); }\n";

static D3D12_RASTERIZER_DESC raster_desc() {
    // The Direct3D 11 viewer's rasterizer state, field for field: solid, no culling (the quad is seen from both sides and
    // the cube's faces are not wound consistently), depth clipping on. Every other field is spelled out rather than
    // left to a zeroed struct, because a zeroed D3D12_RASTERIZER_DESC is not this state.
    D3D12_RASTERIZER_DESC rd{};
    rd.FillMode = D3D12_FILL_MODE_SOLID; rd.CullMode = D3D12_CULL_MODE_NONE; rd.FrontCounterClockwise = FALSE;
    rd.DepthBias = D3D12_DEFAULT_DEPTH_BIAS; rd.DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
    rd.SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS; rd.DepthClipEnable = TRUE;
    rd.MultisampleEnable = FALSE; rd.AntialiasedLineEnable = FALSE; rd.ForcedSampleCount = 0;
    rd.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
    return rd;
}
static D3D12_BLEND_DESC blend_desc(bool alpha) {
    D3D12_BLEND_DESC bd{};
    bd.AlphaToCoverageEnable = FALSE; bd.IndependentBlendEnable = FALSE;
    D3D12_RENDER_TARGET_BLEND_DESC& rt = bd.RenderTarget[0];
    rt.BlendEnable = alpha ? TRUE : FALSE; rt.LogicOpEnable = FALSE;
    rt.SrcBlend = D3D12_BLEND_SRC_ALPHA; rt.DestBlend = D3D12_BLEND_INV_SRC_ALPHA; rt.BlendOp = D3D12_BLEND_OP_ADD;
    rt.SrcBlendAlpha = D3D12_BLEND_ONE; rt.DestBlendAlpha = D3D12_BLEND_ZERO; rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    rt.LogicOp = D3D12_LOGIC_OP_NOOP; rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    return bd;
}
static D3D12_DEPTH_STENCIL_DESC depth_desc(bool on) {
    D3D12_DEPTH_STENCIL_DESC dd{};
    dd.DepthEnable = on ? TRUE : FALSE;
    dd.DepthWriteMask = on ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    dd.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    dd.StencilEnable = FALSE; dd.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK; dd.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
    const D3D12_DEPTH_STENCILOP_DESC keep = { D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS };
    dd.FrontFace = keep; dd.BackFace = keep;
    return dd;
}

// Both pipelines are built into LOCALS and committed only when every step has succeeded, so that key R with a broken shader
// keeps the previous ones and says so rather than leaving the viewer with nothing to draw with.
static bool load_shader(const std::string& path) {
    ID3DBlob* vsb = compile_hlsl_file(path, "VSMain", "vs_5_0");
    if (!vsb) return false;
    ID3DBlob* psb = compile_hlsl_file(path, "PSMain", "ps_5_0");
    if (!psb) { vsb->Release(); return false; }
    ID3DBlob* ovl_vs = nullptr, * ovl_ps = nullptr, * errs = nullptr;
    bool ok = SUCCEEDED(D3DCompile(DEBUG_HLSL, strlen(DEBUG_HLSL), "debug_overlay", nullptr, nullptr, "VSMain", "vs_5_0", 0, 0, &ovl_vs, &errs));
    if (!ok && errs) fprintf(stderr, "OVERLAY VS ERROR:\n%s\n", (const char*)errs->GetBufferPointer());
    safe_release(errs);
    if (ok) {
        ok = SUCCEEDED(D3DCompile(DEBUG_HLSL, strlen(DEBUG_HLSL), "debug_overlay", nullptr, nullptr, "PSMain", "ps_5_0", 0, 0, &ovl_ps, &errs));
        if (!ok && errs) fprintf(stderr, "OVERLAY PS ERROR:\n%s\n", (const char*)errs->GetBufferPointer());
        safe_release(errs);
    }

    ID3D12PipelineState* scene = nullptr, * overlay = nullptr;
    if (ok) {
        const D3D12_INPUT_ELEMENT_DESC scene_il[] = {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        };
        const D3D12_INPUT_ELEMENT_DESC ovl_il[] = {
            { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        };
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = g_rootsig;
        pd.VS.pShaderBytecode = vsb->GetBufferPointer(); pd.VS.BytecodeLength = vsb->GetBufferSize();
        pd.PS.pShaderBytecode = psb->GetBufferPointer(); pd.PS.BytecodeLength = psb->GetBufferSize();
        pd.BlendState = blend_desc(false);
        pd.SampleMask = UINT_MAX;
        pd.RasterizerState = raster_desc();
        pd.DepthStencilState = depth_desc(true);
        pd.InputLayout.pInputElementDescs = scene_il; pd.InputLayout.NumElements = 2;
        pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pd.NumRenderTargets = 1; pd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pd.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        pd.SampleDesc.Count = 1;
        HRESULT hr = g_dev->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&scene));
        if (FAILED(hr)) { fprintf(stderr, "SHADER ERROR: the scene pipeline state could not be created (0x%08X)\n", (unsigned)hr); ok = false; }
        if (ok) {
            pd.VS.pShaderBytecode = ovl_vs->GetBufferPointer(); pd.VS.BytecodeLength = ovl_vs->GetBufferSize();
            pd.PS.pShaderBytecode = ovl_ps->GetBufferPointer(); pd.PS.BytecodeLength = ovl_ps->GetBufferSize();
            pd.BlendState = blend_desc(true);
            pd.DepthStencilState = depth_desc(false);
            pd.InputLayout.pInputElementDescs = ovl_il; pd.InputLayout.NumElements = 2;
            hr = g_dev->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&overlay));
            if (FAILED(hr)) { fprintf(stderr, "SHADER ERROR: the overlay pipeline state could not be created (0x%08X)\n", (unsigned)hr); ok = false; }
        }
    }
    safe_release(vsb); safe_release(psb); safe_release(ovl_vs); safe_release(ovl_ps);
    if (!ok) { safe_release(scene); safe_release(overlay); return false; }
    safe_release(g.pso_scene); safe_release(g.pso_overlay);
    g.pso_scene = scene; g.pso_overlay = overlay;
    printf("shader compiled: %s\n", path.c_str());
#if NNTC_D3D12_LINALG
    // The second scene pipeline, the plan's B.3 step 9. It is built only where the device query passed, and a failure
    // here is NOT a failure of the program: it prints the compiler's or the runtime's own message, leaves g_la_available
    // false and the plain path draws. --linalg 1 turns that into an exit 1 at the one place that decides, in init_common.
    //
    // BOTH STAGES ARE DXC'S. A pipeline whose vertex shader is the fxc DXBC above and whose pixel shader is this DXIL is
    // refused - CreateGraphicsPipelineState returns E_INVALIDARG for that pair, measured here on an RTX 5090 and on the
    // preview WARP - so VSMain is compiled again, from nntc_view_linalg.hlsl, for vs_6_10. That is the answer to the
    // plan's open question B.4. It also means the two pipelines differ by more than the one decode instruction: the
    // vertex stage is a different compiler's, so a last-bit difference in a position could move a pixel of an edge.
    // Separating the compiler from the instruction would need a third arm, the plain shader compiled by DXC, which is
    // the plan's B.6 and is NOT implemented - so the small frame differences recorded in PHASE_B_NOTES.md are what the
    // two pipelines together do, and attributing all of them to fp16 rounding is more than has been shown.
    //
    // --linalg 0 skips the whole of it. The plain path is the product and the run has already said that is the one it
    // wants, so there is nothing for a second pipeline to draw and no reason to spend a run-time DXC compile on it. The
    // query's own verdict is left alone - it says what the DEVICE can do, not what this run asked for - so the overlay
    // still says the plain path on a device that offers the other one, rather than that there is no other one to offer;
    // what key K checks is the pipeline itself, which was never built.
    if (g_la_available && g_la_want != 0) {
        ID3D12PipelineState* linalg = nullptr;
        IDxcBlob* lvs = compile_dxil_file(g.linalg_shader_path, L"VSMain", L"vs_6_10", g_la_fp32, g_la_mul_optimal);
        IDxcBlob* lps = lvs ? compile_dxil_file(g.linalg_shader_path, L"PSMain", L"ps_6_10", g_la_fp32, g_la_mul_optimal) : nullptr;
        if (lps) {
            const D3D12_INPUT_ELEMENT_DESC scene_il2[] = {
                { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
                { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            };
            D3D12_GRAPHICS_PIPELINE_STATE_DESC ld{};
            ld.pRootSignature = g_rootsig;
            ld.VS.pShaderBytecode = lvs->GetBufferPointer(); ld.VS.BytecodeLength = lvs->GetBufferSize();
            ld.PS.pShaderBytecode = lps->GetBufferPointer(); ld.PS.BytecodeLength = lps->GetBufferSize();
            ld.BlendState = blend_desc(false);
            ld.SampleMask = UINT_MAX;
            ld.RasterizerState = raster_desc();
            ld.DepthStencilState = depth_desc(true);
            ld.InputLayout.pInputElementDescs = scene_il2; ld.InputLayout.NumElements = 2;
            ld.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            ld.NumRenderTargets = 1; ld.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            ld.DSVFormat = DXGI_FORMAT_D32_FLOAT;
            ld.SampleDesc.Count = 1;
            const HRESULT lhr = g_dev->CreateGraphicsPipelineState(&ld, IID_PPV_ARGS(&linalg));
            if (FAILED(lhr)) fprintf(stderr, "SHADER ERROR: the linear-algebra pipeline state could not be created (0x%08X)\n", (unsigned)lhr);
        }
        safe_release(lvs); safe_release(lps);
        if (linalg) {
            safe_release(g.pso_linalg);
            g.pso_linalg = linalg;
            printf("shader compiled: %s (ps_6_10, %s bias and result, %s layout)\n", g.linalg_shader_path.c_str(),
                   g_la_fp32 ? "fp32" : "fp16", g_la_mul_optimal ? "multiply-optimal" : "row-major");
        } else if (!g.pso_linalg) {
            g_la_available = false;
            g_la_why_not = "the Shader Model 6.10 pipeline state could not be built (the message above is the compiler's "
                           "or the runtime's own)";
            printf("linear algebra: not available - %s\n", g_la_why_not.c_str());
        } else {
            fprintf(stderr, "Linear-algebra shader reload failed, keeping the previous one.\n");
        }
    }
#endif
    return true;
}
static void reload_shader() {
    // The pipelines cannot be swapped under a frame the GPU is still reading, and this viewer waits for the GPU at the end
    // of every frame anyway; the wait here is what makes the release of the old ones safe on the key-R path as well.
    wait_for_gpu();
    if (!load_shader(g.shader_path)) fprintf(stderr, "Shader reload failed, keeping previous shader.\n");
}

#include "nntc_json.h"   // the JSON reader, in shared/ and shared with bc_check.cpp and the encoder
#include "bc_pack.h"   // the BC4 / BC5 encoder, decoder and level packer (validated by bc_check.cpp against bcdec)
#include "dds.h"   // the DX10 .dds parsing, in shared/ because all three viewers read the same files through it

// Level 0's packed textures from its .dds levels, in bc_pack.h's own file rule: channels 0-1 into a BC5 (a BC4 when that
// is all there is), and what is left into a second texture - a BC5 for a fourth channel, a BC4 for a third one alone. It
// is the same split the encoder writes, so key 4 shows the layout the file format would have given the same plane, and it
// is the same shared/bc_pack.h the other two viewers pack with, so the blocks are the same bytes by construction.
static bool bc_pack_level0(LatentTex& L) {
    const int C = L.C; L.bc_n = bc_level0_files(C);
    if ((L.W % 4) || (L.H % 4)) { fprintf(stderr, "WARNING: level 0 is %dx%d, not a multiple of 4: no BC pack\n", L.W, L.H); L.bc_n = 0; return false; }
    double se_all = 0; size_t n_all = 0; bool lossless = true;
    printf("  level 0 packed at load (key 4 switches to it): ");
    for (int p = 0; p < L.bc_n; p++) {
        const int c0 = 2 * p, nc = bc_level0_file_channels(C, p);
        L.bc_nc[p] = nc;
        if (c0 + nc > L.stored_C) { fprintf(stderr, "\nERROR: level 0 stores %d channels, %d needed for the pack\n", L.stored_C, c0 + nc); L.bc_n = 0; return false; }
        printf("%s%s (channel%s %d%s, %d bit%s)", p ? " + " : "", nc == 1 ? "BC4" : "BC5", nc == 1 ? "" : "s", c0, nc == 1 ? "" : (" and " + std::to_string(c0 + 1)).c_str(), L.bits[c0], L.bits[c0] == 1 ? "" : "s");
        for (int c = 0; c < nc; c++) if (L.bits[c0 + c] > 3) lossless = false;   // every packed channel is one the layout uses
        std::vector<std::vector<uint8_t>> packed((size_t)L.mips);
        std::vector<const uint8_t*> bytes((size_t)L.mips);
        std::vector<size_t> pitch((size_t)L.mips);
        for (int i = 0; i < L.mips; i++) {
            double se = 0; size_t n = 0;
            packed[(size_t)i] = bc_pack_level(L.raw[(size_t)i].data(), L.dims[(size_t)i].first, L.dims[(size_t)i].second, L.stored_C, c0, nc, L.bits, C, se, n);
            {   // The round trip: every packed block decoded on the CPU (the hardware palette) against the exact index value of the source.
                // One decode per BLOCK and then its sixteen texels, rather than a decode per texel, which decoded every block sixteen times.
                const int W = L.dims[(size_t)i].first, H = L.dims[(size_t)i].second, bx = (W + 3) / 4, by = (H + 3) / 4;
                for (int byi = 0; byi < by; byi++) for (int bxi = 0; bxi < bx; bxi++) for (int c = 0; c < nc; c++) {
                    float dec[16]; bc4_decode_block(&packed[(size_t)i][((size_t)byi * bx + bxi) * (nc == 1 ? 8 : 16) + (size_t)c * 8], dec);
                    const int bits_c = L.bits[c0 + c];
                    for (int t = 0; t < 16; t++) {
                        const int x = bxi * 4 + t % 4, y = byi * 4 + t / 4; if (x >= W || y >= H) continue;
                        const int k = L.raw[(size_t)i][((size_t)y * W + x) * L.stored_C + c0 + c] >> (8 - bits_c);
                        L.bc_maxerr = std::max(L.bc_maxerr, std::fabs(dec[t] - k * 255.0 / ((1 << bits_c) - 1)));
                    }
                }
            }
            bytes[(size_t)i] = packed[(size_t)i].data();
            pitch[(size_t)i] = (size_t)((L.dims[(size_t)i].first + 3) / 4) * (nc == 1 ? 8 : 16);
            se_all += se; n_all += n;
        }
        L.bc_tex[p] = create_texture_with_levels(nc == 1 ? DXGI_FORMAT_BC4_UNORM : DXGI_FORMAT_BC5_UNORM, L.W, L.H, L.mips, bytes, pitch, "the level-0 BC pack");
        if (!L.bc_tex[p]) {
            fprintf(stderr, "\nERROR: the BC texture creation failed\n");
            for (int q = 0; q < 2; q++) safe_release(L.bc_tex[q]);   // the first texture too, when it is the second that failed
            L.bc_n = 0; return false;
        }
    }
    L.bc_lossless = lossless; L.bc_psnr = n_all && se_all > 0 ? 10.0 * std::log10(255.0 * 255.0 / (se_all / (double)n_all)) : 0.0;
    if (lossless) printf(": lossless (0 and 255 as the endpoints, the index in the selectors; the round trip's largest error %.3g / 255 in float)\n", L.bc_maxerr); else printf(": lossy, packing PSNR %.2f dB over the chain against the exact index values (largest error %.1f / 255)\n", L.bc_psnr, L.bc_maxerr);
    return true;
}

// ---------------------------------------------------------------------------
// The DX10 .dds loader (docs/FORMAT.md section 2): the 148-byte header, the levels base-first and tightly packed, into a
// committed mipmapped texture in the default heap. Uncompressed R8 / R8G8 / R8G8B8A8, and block-compressed BC4_UNORM (80) /
// BC5_UNORM (83), which is how the encoder writes level 0 by default. `second` loads the SECOND file of a two-file level 0
// into the same LatentTex, which is bound to the slot the load-time pack already uses for channels 2-3.
//
// The parsing itself is shared/dds.h, which the Direct3D 11 viewer's own header walk was lifted into so that every viewer
// reads the same bytes through the same checks; what is left here is the Direct3D 12 tail.
// ---------------------------------------------------------------------------
static bool load_dds(const std::string& path, LatentTex& L, bool second = false) {
    DdsImage img;
    if (!dds_read(path, img)) return false;
    const int W = img.W, H = img.H, nmip = img.mips, dxgi = img.dxgi, C = img.channels;
    const bool bc = img.bc;
    // The two files of one level share the extent and the level count, and NOT the format: a three-channel level 0 is a
    // BC5 of channels 0-1 and a BC4 of channel 2, so only that both are block-compressed (or neither is) is required.
    if (second && (W != L.W || H != L.H || nmip != L.mips)) { fprintf(stderr, "ERROR: '%s' does not match the first file of this level (%dx%d, %d level%s)\n", path.c_str(), L.W, L.H, L.mips, L.mips == 1 ? "" : "s"); return false; }
    if (second && bc != L.file_bc) { fprintf(stderr, "ERROR: '%s' is %s where the first file of this level is not\n", path.c_str(), bc ? "block-compressed" : "uncompressed"); return false; }
    std::vector<const uint8_t*> bytes((size_t)nmip);
    std::vector<size_t> pitch((size_t)nmip);
    std::vector<std::pair<int, int>> level_dim((size_t)nmip);
    for (int i = 0; i < nmip; i++) {
        bytes[(size_t)i] = (const uint8_t*)img.bytes.data() + img.levels[(size_t)i].offset;
        pitch[(size_t)i] = (size_t)img.levels[(size_t)i].pitch;
        level_dim[(size_t)i] = { img.levels[(size_t)i].w, img.levels[(size_t)i].h };
    }
    ID3D12Resource* tex = create_texture_with_levels((DXGI_FORMAT)dxgi, W, H, nmip, bytes, pitch, path.c_str());
    if (!tex) return false;
    // A second file replacing a first one that is already there releases it: the two are loaded in sequence, and a
    // failure between them used to leave the first leaked.
    if (second) { safe_release(L.tex_b); L.tex_b = tex; L.stored_C_b = C; L.dxgi_b = dxgi; }
    else {
        safe_release(L.tex);
        L.tex = tex; L.W = W; L.H = H; L.stored_C = C; L.mips = nmip; L.dxgi = dxgi; L.file = path; L.file_bc = bc;
        L.raw.clear(); L.dims = level_dim;
        if (!bc) { L.raw.resize((size_t)nmip); for (int i = 0; i < nmip; i++) L.raw[(size_t)i].assign(img.bytes.begin() + img.levels[(size_t)i].offset, img.bytes.begin() + img.levels[(size_t)i].offset + img.levels[(size_t)i].size); }   // kept for the pack
    }
    printf("  %s: %dx%d, %d mip level%s, DXGI format %d (%s, %d channel%s stored)\n", path.c_str(), W, H, nmip, nmip == 1 ? "" : "s", dxgi, bc ? (dxgi == 80 ? "BC4_UNORM" : "BC5_UNORM") : "uncompressed", C, C == 1 ? "" : "s");
    return true;
}

// ---------------------------------------------------------------------------
// The asset: PREFIX_nntc.json, then the two .dds files it names (next to it), then the decoder constants. The path is
// taken as given - the viewer derives no name of its own - so an asset written under any descriptor name loads.
// ---------------------------------------------------------------------------
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
    const size_t slash = json_path.find_last_of("/\\"); const std::string dir = slash == std::string::npos ? "" : json_path.substr(0, slash + 1);
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
        const JVal& t = texs->arr[(size_t)l]; LatentTex& L = g.lat[l];
        if ((int)t.number("level", -1) != l) { fprintf(stderr, "ERROR: textures[%d] is not level %d\n", l, l); return false; }
        L.C = (int)t.number("channels_used");
        // bits_per_channel decides a shift (byte >> (8 - bits)) and a divisor (2^bits - 1), so 0 or a value above 8
        // would be a negative shift - undefined - or a division by zero. The entries are checked for being numbers at
        // all, too: a JSON that carried a string there would otherwise read as .num = 0.
        const JVal* bits = t.get("bits_per_channel");
        for (int c = 0; c < 4; c++) L.bits[c] = bits && bits->kind == JVal::ARR && c < (int)bits->arr.size() && bits->arr[(size_t)c].kind == JVal::NUM ? (int)bits->arr[(size_t)c].num : 8;
        for (int c = 0; c < 4; c++) {
            const int lo_bits = l == 0 ? 1 : 4;   // level 0 is written at 8 bits under --l0 bc8 (the default) and at 1-4 under --l0 palette; level 1 at 4-8
            if (L.bits[c] < lo_bits || L.bits[c] > 8) { fprintf(stderr, "ERROR: level %d: bits_per_channel[%d] is %d, which is outside %d..8\n", l, c, L.bits[c], lo_bits); return false; }
        }
        // "file" names one .dds and "files" names two, which is how a block-compressed level 0 of three or four channels is
        // stored. Each entry of "files" is an OBJECT carrying that file's own dxgi_format_id and channels_stored, because the
        // two need not agree; an entry that is a BARE STRING is the older spelling of the same key and is read as a name with
        // nothing to cross-check, which is what every other reader in this tree does.
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
                want_dxgi[i] = e.kind == JVal::OBJ ? (int)e.number("dxgi_format_id", 0) : 0;      // 0 = the older spelling: nothing published to cross-check
                want_stored[i] = e.kind == JVal::OBJ ? (int)e.number("channels_stored", 0) : 0;
            }
            if (!load_dds(beside(names[0]), L) || !load_dds(beside(names[1]), L, true)) return false;
            L.file_files = 2;
            const int got_dxgi[2] = { L.dxgi, L.dxgi_b }, got_stored[2] = { L.stored_C, L.stored_C_b };
            for (int i = 0; i < 2; i++)
                if ((want_dxgi[i] && want_dxgi[i] != got_dxgi[i]) || (want_stored[i] && want_stored[i] != got_stored[i])) {
                    fprintf(stderr, "ERROR: level %d: the JSON says %s is DXGI %d storing %d channel(s) and the file is DXGI %d storing %d\n",
                            l, names[i].c_str(), want_dxgi[i], want_stored[i], got_dxgi[i], got_stored[i]);
                    return false;
                }
        } else if (!load_dds(beside(t.string("file")), L)) return false;
        // The channels the level's files hold between them: one file stores stored_C, two store stored_C + stored_C_b (2 + 1 for a
        // three-channel BC level 0, which is exactly the three the decoder reads).
        const int stored_all = L.stored_C + (L.file_files > 1 ? L.stored_C_b : 0);
        if (L.C < 1 || L.C > 4 || L.C > stored_all) { fprintf(stderr, "ERROR: level %d: %d channels used of %d stored\n", l, L.C, stored_all); return false; }
        // The load-time pack, for an uncompressed level 0 only; key 4 binds it. It is not fatal when it fails - the
        // file's own plane is still there and is what the viewer shows - but it says so rather than leaving key 4 dead
        // with no explanation, and the raw levels it needed are dropped once it has run.
        if (l == 0 && !L.file_bc) {
            if (!bc_pack_level0(L)) fprintf(stderr, "WARNING: level 0 was not packed at load; key 4 has nothing to bind\n");
            L.raw.clear(); L.raw.shrink_to_fit();
        }
        // Level 1's LOD shift (key L). The value is the JSON's own lod_bias_level1, which is what the format says the
        // level-1 sample must carry; log2(block) is only the value the writer puts there, and a file that ever said
        // something else would mean it. Present and not a number is refused, as an out-of-range number is.
        if (l == 1) {
            const JVal* lbv = root.get("lod_bias_level1");
            if (lbv && lbv->kind != JVal::NUM) { fprintf(stderr, "ERROR: the descriptor's lod_bias_level1 is not a number\n"); return false; }
            const float lb = lbv ? (float)lbv->num : std::log2((float)(g.block > 1 ? g.block : 1));
            if (!check_level1_lod_shift(g.block, lb)) return false;
            g.lod_bias_level1 = lb;
        }
        const JVal* dq = t.get("dequantise"); const JVal* lo = dq ? dq->get("lo") : nullptr; const JVal* hi = dq ? dq->get("hi") : nullptr;
        if (!lo || !hi || lo->kind != JVal::ARR || hi->kind != JVal::ARR || (int)lo->arr.size() < L.C || (int)hi->arr.size() < L.C) { fprintf(stderr, "ERROR: level %d: no lo / hi per channel\n", l); return false; }
        for (int c = 0; c < L.C; c++) { L.lo[c] = (float)lo->arr[(size_t)c].num; L.hi[c] = (float)hi->arr[(size_t)c].num; }
        printf("  level %d: %d channel%s used, bits", l, L.C, L.C == 1 ? "" : "s"); for (int c = 0; c < L.C; c++) printf(" %d", L.bits[c]);
        printf(", dequantise value = lo + sample * (hi - lo) with lo / hi"); for (int c = 0; c < L.C; c++) printf(" [%g, %g]", L.lo[c], L.hi[c]); printf("\n");
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
    DecCB& d = g.dec; memset(&d, 0, sizeof(d));
    for (int c = 0; c < 4; c++) { d.lo0[c] = g.lat[0].lo[c]; d.hi0[c] = g.lat[0].hi[c]; d.lo1[c] = g.lat[1].lo[c]; d.hi1[c] = g.lat[1].hi[c]; }
    d.dims[0] = C0; d.dims[1] = C1; d.dims[2] = nin; d.dims[3] = nout; d.sel[0] = mask; d.sel[1] = 0;
    // W row-major into the shader's six float4 per row, and the bias into five. This is also the packing the plan's phase B
    // reads its padded 18 by 24 fp16 matrix out of, which is why it is one loop and not folded into the parse above.
    for (int r = 0; r < nout; r++) for (int c = 0; c < nin; c++) d.W[r * 6 + c / 4][c % 4] = (float)Wv->arr[(size_t)r * nin + c].num;
    for (int r = 0; r < nout; r++) d.bias[r / 4][r % 4] = (float)bv->arr[(size_t)r].num;
    printf("  decoder: bilinear, terms '%s', phi %d -> %d outputs (%d texture%s), %d weights\n", g.terms.c_str(), nin, nout, g.textures_out, g.textures_out == 1 ? "" : "s", nout * nin + nout);
    return true;
}

// ---------------------------------------------------------------------------
// The descriptors. Both three-texture sets are written once, after the asset is loaded, and the frame picks between them
// with one SetGraphicsRootDescriptorTable: t0 level 0, t1 level 1, t2 level 0's channels past the first two when they live
// in a texture of their own - which is the case both when the file is two of them and when the load-time pack made two out
// of an uncompressed level 0.
//
// A slot with no texture gets a NULL descriptor rather than nothing at all. Direct3D 11 accepts a null SRV pointer and
// Direct3D 12 does not: an unwritten descriptor is whatever the heap's memory held, and the shader does not sample the slot
// (sel.z says how many textures level 0 is read from) but the debug layer reads the whole table.
// ---------------------------------------------------------------------------
static void write_srv(int index, ID3D12Resource* tex, DXGI_FORMAT fmt, int mips) {
    D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
    sd.Format = tex ? fmt : DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sd.Texture2D.MostDetailedMip = 0; sd.Texture2D.MipLevels = (UINT)(tex ? mips : 1);
    sd.Texture2D.PlaneSlice = 0; sd.Texture2D.ResourceMinLODClamp = 0.0f;
    g_dev->CreateShaderResourceView(tex, &sd, cpu_handle(g_srv_heap, g_srv_step, index));
}
#if NNTC_D3D12_LINALG
// The weights buffer's descriptor: a RAW view (R32_TYPELESS with the RAW flag), which is what a ByteAddressBuffer binds
// to. A null resource here is a valid descriptor and is what the slot holds until the buffer exists, so that a table
// bound before the asset is read has nothing undefined in it.
static void write_raw_srv(int index, ID3D12Resource* buf, UINT64 bytes) {
    D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
    sd.Format = DXGI_FORMAT_R32_TYPELESS;
    sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sd.Buffer.FirstElement = 0;
    sd.Buffer.NumElements = (UINT)((buf ? bytes : 4) / 4);
    sd.Buffer.StructureByteStride = 0;
    sd.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    g_dev->CreateShaderResourceView(buf, &sd, cpu_handle(g_srv_heap, g_srv_step, index));
}
static ID3D12Resource* g_la_buffer = nullptr;   // the fp16 matrix and the bias, in one default-heap buffer
static UINT64          g_la_bytes = 0;
#endif

static void refresh_srv_sets() {
    const LatentTex& L0 = g.lat[0]; const LatentTex& L1 = g.lat[1];
    write_srv(SRV_SET_PLAIN + 0, L0.tex, (DXGI_FORMAT)L0.dxgi, L0.mips);
    write_srv(SRV_SET_PLAIN + 1, L1.tex, (DXGI_FORMAT)L1.dxgi, L1.mips);
    write_srv(SRV_SET_PLAIN + 2, L0.tex_b, (DXGI_FORMAT)L0.dxgi_b, L0.mips);
    // The pack's second texture is a BC5 for a fourth channel and a BC4 for a third one alone; the shader reads .rg of it
    // either way and uses only .r when C0 is 3, and a BC4 SRV returns (r, 0, 0, 1), so the green it would read is never
    // part of phi.
    write_srv(SRV_SET_PACKED + 0, L0.bc_n > 0 ? L0.bc_tex[0] : nullptr, L0.bc_nc[0] == 1 ? DXGI_FORMAT_BC4_UNORM : DXGI_FORMAT_BC5_UNORM, L0.mips);
    write_srv(SRV_SET_PACKED + 1, L1.tex, (DXGI_FORMAT)L1.dxgi, L1.mips);
    write_srv(SRV_SET_PACKED + 2, L0.bc_n > 1 ? L0.bc_tex[1] : nullptr, L0.bc_nc[1] == 1 ? DXGI_FORMAT_BC4_UNORM : DXGI_FORMAT_BC5_UNORM, L0.mips);
#if NNTC_D3D12_LINALG
    // t3 in all three sets: the same buffer, because nothing about the weights depends on which level-0 textures are
    // bound, and the overlay's own set needs a valid fourth descriptor whether its pipeline reads it or not.
    for (int s = 0; s < 3; s++) write_raw_srv(s * SRV_PER_SET + 3, g_la_buffer, g_la_bytes);
#endif
}

// ---------------------------------------------------------------------------
// Geometry (position float3 + uv float2, interleaved), in upload heaps.
//
// A default heap and a copy would be the usual shape for geometry that never changes, and it would buy nothing here: these
// are a few hundred bytes read once per frame by a viewer whose whole frame waits on the GPU anyway, and an upload heap is
// one resource instead of three.
// ---------------------------------------------------------------------------
static ID3D12Resource* g_quad_vb = nullptr, * g_quad_ib = nullptr;
static ID3D12Resource* g_cube_vb = nullptr, * g_cube_ib = nullptr;
static D3D12_VERTEX_BUFFER_VIEW g_quad_vbv{}, g_cube_vbv{};
static D3D12_INDEX_BUFFER_VIEW  g_quad_ibv{}, g_cube_ibv{};
static int g_cube_index_count = 0;

static ID3D12Resource* make_upload_buffer(const void* data, size_t bytes) {
    const D3D12_HEAP_PROPERTIES up = heap_props(D3D12_HEAP_TYPE_UPLOAD);
    const D3D12_RESOURCE_DESC rd = buffer_desc((UINT64)bytes);
    ID3D12Resource* b = nullptr;
    if (FAILED(g_dev->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&b)))) return nullptr;
    if (data) {
        uint8_t* p = nullptr; D3D12_RANGE none{ 0, 0 };
        if (FAILED(b->Map(0, &none, (void**)&p))) { b->Release(); return nullptr; }
        memcpy(p, data, bytes);
        b->Unmap(0, nullptr);
    }
    return b;
}
static bool create_quad(float aspect) {
    float hw, hh;
    if (aspect >= 1.0f) { hw=1.0f; hh=1.0f/aspect; } else { hw=aspect; hh=1.0f; }
    // UV origin: (0,0) = top-left = the first uploaded row; the .dds levels are top-row-first, so v runs 0..1 top to bottom with no flip.
    float v[] = {
        -hw,-hh,0.0f, 0.0f,1.0f,  hw,-hh,0.0f, 1.0f,1.0f,
         hw, hh,0.0f, 1.0f,0.0f, -hw, hh,0.0f, 0.0f,0.0f,
    };
    uint32_t idx[] = {0,1,2, 0,2,3};
    g_quad_vb = make_upload_buffer(v, sizeof(v));
    g_quad_ib = make_upload_buffer(idx, sizeof(idx));
    if (!g_quad_vb || !g_quad_ib) return false;
    g_quad_vbv.BufferLocation = g_quad_vb->GetGPUVirtualAddress(); g_quad_vbv.SizeInBytes = (UINT)sizeof(v); g_quad_vbv.StrideInBytes = 5 * sizeof(float);
    g_quad_ibv.BufferLocation = g_quad_ib->GetGPUVirtualAddress(); g_quad_ibv.SizeInBytes = (UINT)sizeof(idx); g_quad_ibv.Format = DXGI_FORMAT_R32_UINT;
    return true;
}
static bool create_cube() {
    const float h=0.5f;
    float v[] = {
        -h,-h, h, 0,1,  h,-h, h, 1,1,  h, h, h, 1,0, -h, h, h, 0,0,
         h,-h,-h, 0,1, -h,-h,-h, 1,1, -h, h,-h, 1,0,  h, h,-h, 0,0,
         h,-h, h, 0,1,  h,-h,-h, 1,1,  h, h,-h, 1,0,  h, h, h, 0,0,
        -h,-h,-h, 0,1, -h,-h, h, 1,1, -h, h, h, 1,0, -h, h,-h, 0,0,
        -h, h, h, 0,1,  h, h, h, 1,1,  h, h,-h, 1,0, -h, h,-h, 0,0,
        -h,-h,-h, 0,1,  h,-h,-h, 1,1,  h,-h, h, 1,0, -h,-h, h, 0,0,
    };
    std::vector<uint32_t> idx;
    for (int i=0;i<6;i++){ int b=i*4; idx.insert(idx.end(),{(uint32_t)b,(uint32_t)b+1,(uint32_t)b+2,(uint32_t)b,(uint32_t)b+2,(uint32_t)b+3}); }
    g_cube_vb = make_upload_buffer(v, sizeof(v));
    g_cube_ib = make_upload_buffer(idx.data(), idx.size()*sizeof(uint32_t));
    if (!g_cube_vb || !g_cube_ib) return false;
    g_cube_vbv.BufferLocation = g_cube_vb->GetGPUVirtualAddress(); g_cube_vbv.SizeInBytes = (UINT)sizeof(v); g_cube_vbv.StrideInBytes = 5 * sizeof(float);
    g_cube_ibv.BufferLocation = g_cube_ib->GetGPUVirtualAddress(); g_cube_ibv.SizeInBytes = (UINT)(idx.size()*sizeof(uint32_t)); g_cube_ibv.Format = DXGI_FORMAT_R32_UINT;
    g_cube_index_count = (int)idx.size();
    return true;
}

#if NNTC_D3D12_LINALG
// ---------------------------------------------------------------------------
// THE WEIGHTS BUFFER: the padded 18 by 24 matrix and the 18-entry bias, once, at load (the plan's B.5).
//
// The TYPE conversion is done HERE, on the host, and the LAYOUT conversion on the GPU. ConvertLinearAlgebraMatrix can do
// both, and splitting them is deliberate: fp32 to fp16 is exact arithmetic on numbers this program is holding anyway, it
// is the same float_to_half the Vulkan viewer uses so the two accelerated paths quantise the weights identically, and it
// keeps the range guard below on the side of the fence where the numbers are. What is left for the GPU is the one thing
// only the device can do - fp16 row-major into its own multiply-optimal layout - and that step is optional.
// ---------------------------------------------------------------------------
static uint16_t float_to_half(float f) {
    uint32_t bits = 0;
    memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exponent = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mantissa = bits & 0x7FFFFFu;
    if (((bits >> 23) & 0xFFu) == 0xFFu)                      // an infinity or a NaN stays one
        return (uint16_t)(sign | 0x7C00u | (mantissa ? 0x200u : 0u));
    if (exponent >= 0x1F) return (uint16_t)(sign | 0x7C00u);  // too large for fp16: infinity
    if (exponent <= 0) {
        if (exponent < -10) return (uint16_t)sign;            // too small even to be subnormal
        const uint32_t m = mantissa | 0x800000u;              // the implicit one, then shifted into subnormal
        const int shift = 14 - exponent;
        const uint32_t half = m >> shift;
        const uint32_t rest = m & ((1u << shift) - 1u), halfway = 1u << (shift - 1);
        return (uint16_t)(sign | (half + ((rest > halfway || (rest == halfway && (half & 1u))) ? 1u : 0u)));
    }
    const uint32_t half = ((uint32_t)exponent << 10) | (mantissa >> 13);
    const uint32_t rest = mantissa & 0x1FFFu;
    return (uint16_t)(sign | (half + ((rest > 0x1000u || (rest == 0x1000u && (half & 1u))) ? 1u : 0u)));
}

// A default-heap buffer filled once from a staging upload recorded on the load-time command list. `flags` carries
// ALLOW_UNORDERED_ACCESS where the conversion needs it, and `state` is where the buffer is left.
static ID3D12Resource* make_default_buffer(const void* data, size_t bytes, size_t offset, size_t total,
                                           D3D12_RESOURCE_FLAGS flags, const char* what) {
    D3D12_RESOURCE_DESC rd = buffer_desc((UINT64)total);
    rd.Flags = flags;
    const D3D12_HEAP_PROPERTIES def = heap_props(D3D12_HEAP_TYPE_DEFAULT);
    ID3D12Resource* buf = nullptr;
    HRESULT hr = g_dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&buf));
    if (FAILED(hr)) { fprintf(stderr, "ERROR: %s could not be created (0x%08X)\n", what, (unsigned)hr); return nullptr; }
    ID3D12Resource* staging = make_upload_buffer(nullptr, total);
    if (!staging) { fprintf(stderr, "ERROR: the staging buffer for %s could not be created\n", what); buf->Release(); return nullptr; }
    uint8_t* p = nullptr; D3D12_RANGE none{ 0, 0 };
    if (FAILED(staging->Map(0, &none, (void**)&p))) { fprintf(stderr, "ERROR: the staging buffer for %s could not be mapped\n", what); staging->Release(); buf->Release(); return nullptr; }
    memset(p, 0, total);
    if (data) memcpy(p + offset, data, bytes);
    staging->Unmap(0, nullptr);
    g_list->CopyBufferRegion(buf, 0, staging, 0, (UINT64)total);
    g_upload_keepalive.push_back(staging);
    return buf;
}

// A step of this path that a resource or an interface refused. It is the range guard's shape below, written once because
// four steps need it: the reason is printed in the words every other refusal on this path uses, the path is turned off so
// the plain one draws, and only --linalg 1 - the run asking for this path by name - turns it into an exit, which it does
// in init_common, on the false this returns.
static bool linalg_step_failed(const char* why) {
    g_la_why_not = why;
    g_la_available = false;
    printf("linear algebra: not available - %s\n", g_la_why_not.c_str());
    if (g_la_want == 1) {
        fprintf(stderr, "ERROR: --linalg 1 was asked for and this run cannot: %s\n", g_la_why_not.c_str());
        return false;
    }
    return true;   // the plain path draws it
}

// The whole of it. It runs on the already-open load-time command list, between load_asset and the one flush, so the
// copies and the conversion below are in the same submission as every texture upload and are waited for once.
static bool build_linalg_weights() {
    // Nothing to build where the query did not pass, and nothing to build where the run asked for the plain path: under
    // --linalg 0 no second pipeline exists to read this buffer, so it is not allocated and no conversion is recorded.
    // The query's verdict is left as it is, because it is what the overlay and key K read.
    if (!g_la_available || g_la_want == 0) return true;

    // The padded matrix and bias, out of the same DecoderConstants the plain shader reads: the rows past nout and the
    // columns past nin are the zeros load_asset's memset left there, which is what makes the padded product exact.
    const DecCB& d = g.dec;
    std::vector<float> src((size_t)LA_M * LA_K, 0.0f);
    for (UINT r = 0; r < LA_M; r++)
        for (UINT c = 0; c < LA_K; c++) src[(size_t)r * LA_K + c] = d.W[r * 6 + c / 4][c % 4];
    std::vector<float> bias(LA_M, 0.0f);
    for (UINT r = 0; r < LA_M; r++) bias[r] = d.bias[r / 4][r % 4];

    // THE RANGE GUARD, and it is the row sum rather than the largest weight. Every row this viewer can run reads the
    // matrix as fp16, and under the all-fp16 row the products, the running sum and the result are fp16 too, so a decode
    // can overflow to an infinity - a picture that is wrong rather than slightly wrong - with every stored weight
    // comfortably inside 65504. The bound is the one the plan's B.5 gives:
    //
    //     sum over j of |W_rj| * max|phi_j|  +  |b_r|  <=  65504
    //
    // with max|phi_j| taken from the asset itself: a level-0 or level-1 feature is a dequantised sample, so |z| is at
    // most max(|lo|, |hi|) of its channel, and a product feature is bounded by the product of its two channels' bounds.
    // It is exact arithmetic on numbers already in hand, so it costs nothing and refuses only what would overflow. (The
    // small end needs no guard: an fp16 subnormal reaches 6e-8 and anything under that becomes a zero, which is a
    // rounding error of the size the two paths' frames are compared at.)
    float phi_max[LA_K] = { 0.0f };
    {
        const int C0 = d.dims[0], C1 = d.dims[1], mask = d.sel[0];
        float b0[4] = { 0, 0, 0, 0 }, b1[4] = { 0, 0, 0, 0 };
        for (int c = 0; c < 4; c++) {
            b0[c] = std::max(std::fabs(d.lo0[c]), std::fabs(d.hi0[c]));
            b1[c] = std::max(std::fabs(d.lo1[c]), std::fabs(d.hi1[c]));
        }
        int k = 0;
        if (mask & 1) for (int j = 0; j < 4; j++) if (j < C1 && k < (int)LA_K) phi_max[k++] = b1[j];                       // [a]  c_j
        if (mask & 2) for (int i = 0; i < 4; i++) if (i < C0 && k < (int)LA_K) phi_max[k++] = b0[i];                       // [b]  s_i
        if (mask & 4) for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) if (i < C0 && j < C1 && k < (int)LA_K) phi_max[k++] = b0[i] * b1[j];   // [sc] s_i c_j
    }
    float worst = 0.0f; int worst_row = 0; float largest_stored = 0.0f;
    for (float v : src) largest_stored = std::max(largest_stored, std::fabs(v));
    for (float v : bias) largest_stored = std::max(largest_stored, std::fabs(v));
    for (UINT r = 0; r < LA_M; r++) {
        double sum = std::fabs(bias[r]);
        for (UINT c = 0; c < LA_K; c++) sum += (double)std::fabs(src[(size_t)r * LA_K + c]) * (double)phi_max[c];
        if ((float)sum > worst) { worst = (float)sum; worst_row = (int)r; }
    }
    if (worst > 65504.0f || largest_stored > 65504.0f) {
        char why[288];
        if (largest_stored > 65504.0f)
            snprintf(why, sizeof(why), "this asset's decoder has a weight of %g, past fp16's largest finite value of "
                                       "65504, and every row this viewer can use reads the matrix as fp16", (double)largest_stored);
        else
            snprintf(why, sizeof(why), "output row %d of this asset's decoder can reach %g in magnitude (sum of |W| times "
                                       "the largest |phi| the dequantisation allows, plus |bias|), past fp16's largest "
                                       "finite value of 65504", worst_row, (double)worst);
        g_la_why_not = why;
        g_la_available = false;
        printf("linear algebra: not available - %s\n", g_la_why_not.c_str());
        if (g_la_want == 1) {
            fprintf(stderr, "ERROR: --linalg 1 was asked for and this asset cannot: %s\n", g_la_why_not.c_str());
            return false;
        }
        return true;   // the plain path draws it, and reads W as the fp32 it is
    }

    std::vector<uint16_t> matrix16((size_t)LA_M * LA_K, 0);
    for (size_t i = 0; i < matrix16.size(); i++) matrix16[i] = float_to_half(src[i]);
    // The bias is read according to the row's bias type and nothing converts it on the way in, so it goes into the
    // buffer in that type: fp32 under the preferred row, fp16 under the mandatory one.
    std::vector<uint16_t> bias16(LA_M, 0);
    for (UINT r = 0; r < LA_M; r++) bias16[r] = float_to_half(bias[r]);
    const void* bias_bytes = g_la_fp32 ? (const void*)bias.data() : (const void*)bias16.data();
    const size_t bias_size = g_la_fp32 ? bias.size() * sizeof(float) : bias16.size() * sizeof(uint16_t);
    const size_t row_bytes = (size_t)LA_K * sizeof(uint16_t);          // 48, already a multiple of the 16 the spec wants
    const size_t matrix_bytes = (size_t)LA_M * row_bytes;              // 864

    UINT dest_size = 0;
    ID3D12DevicePreview* devp = nullptr;
    if (g_la_mul_optimal && SUCCEEDED(g_dev->QueryInterface(IID_PPV_ARGS(&devp)))) {
        D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_DEST_INFO info{};
        info.DestLayout = D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_MUL_OPTIMAL;
        info.DestStride = 0;                 // an INPUT field, and 0 is what an optimal layout takes: its addressing is opaque
        info.NumRows = LA_M; info.NumColumns = LA_K;
        info.DestDataType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
        devp->GetLinearAlgebraMatrixConversionDestinationInfo(&info);
        dest_size = info.DestSize;           // the one thing this query answers; DestStride is never read back out of it
        if (!dest_size) {
            fprintf(stderr, "WARNING: the device asked for a zero-byte multiply-optimal matrix; the row-major layout is used instead\n");
            g_la_mul_optimal = false;
        }
    } else if (g_la_mul_optimal) {
        fprintf(stderr, "WARNING: ID3D12DevicePreview is not available on this device; the row-major layout is used instead\n");
        g_la_mul_optimal = false;
    }
    safe_release(devp);
    if (!g_la_mul_optimal) dest_size = (UINT)matrix_bytes;
    // The stride, decided HERE and not before, because either branch above can fall back to row-major and a stride left
    // over from the optimal layout would make the shader's RowMajor load read row 0 for every row. It is the application's
    // own number either way: 0 for the multiply-optimal layout, whose addressing is the device's and whose stride is
    // ignored, and one row of 24 fp16 for the row-major one this program writes.
    const UINT dest_stride = g_la_mul_optimal ? 0u : (UINT)row_bytes;

    // The layout of the one buffer. The matrix starts at byte 0, which is the base of a committed resource and so is far
    // past the 128 bytes a thread-scope load wants; the bias starts at the next 128-byte boundary after the matrix, which
    // is this program's conservatism where the specification is silent about the bias; and the whole thing is rounded up
    // to 128, because a long-vector load of eighteen entries is free to read in wider units than the bias's own length
    // and a buffer that ended exactly at it would let the last of those run past the allocation.
    const UINT64 bias_offset = ((UINT64)dest_size + 127) & ~(UINT64)127;
    g_la_bytes = (bias_offset + bias_size + 127) & ~(UINT64)127;

    if (!g_la_mul_optimal) {
        // The simple form, and the one that needs no conversion API at all: the specification permits a thread-scope
        // load from RowMajor, so the fp16 matrix goes into the buffer as the host wrote it and the bias after it.
        std::vector<uint8_t> whole((size_t)g_la_bytes, 0);
        memcpy(whole.data(), matrix16.data(), matrix_bytes);
        memcpy(whole.data() + bias_offset, bias_bytes, bias_size);
        g_la_buffer = make_default_buffer(whole.data(), whole.size(), 0, (size_t)g_la_bytes, D3D12_RESOURCE_FLAG_NONE, "the linear-algebra weights buffer");
        if (!g_la_buffer) return linalg_step_failed("the linear-algebra weights buffer could not be created (the message above is the runtime's own)");
        barrier(g_la_buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    } else {
        // The multiply-optimal form. Unlike the Vulkan counterpart, which converts on the host, this is a GPU command on
        // two GPU virtual addresses, so both buffers are default-heap ones and the states are the spec's: the source is
        // read as a non-pixel shader resource and the destination is written as an unordered access.
        ID3D12GraphicsCommandListPreview* listp = nullptr;
        if (FAILED(g_list->QueryInterface(IID_PPV_ARGS(&listp)))) {
            return linalg_step_failed("ID3D12GraphicsCommandListPreview is not available on this command list, so the "
                                      "matrix cannot be converted into the device's multiply-optimal layout");
        }
        ID3D12Resource* rowmajor = make_default_buffer(matrix16.data(), matrix_bytes, 0, matrix_bytes, D3D12_RESOURCE_FLAG_NONE, "the row-major source matrix");
        if (!rowmajor) { safe_release(listp); return linalg_step_failed("the row-major source matrix could not be created (the message above is the runtime's own)"); }
        barrier(rowmajor, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        // The bias is copied in FIRST, while the destination is still a copy destination; the conversion writes only the
        // DestSize bytes at the base and the bias sits past them, so the two do not overlap.
        g_la_buffer = make_default_buffer(bias_bytes, bias_size, (size_t)bias_offset, (size_t)g_la_bytes,
                                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, "the linear-algebra weights buffer");
        // The source goes to the keepalive rather than straight back: the load-time list already carries a copy into it
        // and that list is still submitted when the plain path draws, so releasing it here would leave that copy writing
        // into a freed resource.
        if (!g_la_buffer) { g_upload_keepalive.push_back(rowmajor); safe_release(listp); return linalg_step_failed("the linear-algebra weights buffer could not be created (the message above is the runtime's own)"); }
        barrier(g_la_buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_INFO conv{};
        conv.DestInfo.DestSize = dest_size;
        conv.DestInfo.DestLayout = D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_MUL_OPTIMAL;
        conv.DestInfo.DestStride = dest_stride;
        conv.DestInfo.NumRows = LA_M; conv.DestInfo.NumColumns = LA_K;
        conv.DestInfo.DestDataType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
        conv.SrcInfo.SrcSize = (UINT)matrix_bytes;
        conv.SrcInfo.SrcDataType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;   // a LAYOUT conversion only: the type is the host's work
        conv.SrcInfo.SrcLayout = D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_ROW_MAJOR;
        conv.SrcInfo.SrcStride = (UINT)row_bytes;
        conv.DataDesc.DestVA = g_la_buffer->GetGPUVirtualAddress();
        conv.DataDesc.SrcVA = rowmajor->GetGPUVirtualAddress();
        listp->ConvertLinearAlgebraMatrix(&conv, 1);
        safe_release(listp);

        D3D12_RESOURCE_BARRIER uav{};
        uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; uav.UAV.pResource = g_la_buffer;
        g_list->ResourceBarrier(1, &uav);
        barrier(g_la_buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        g_upload_keepalive.push_back(rowmajor);   // released with the staging buffers, once the flush has completed
    }

    g.dec.linalg[0] = (uint32_t)bias_offset;
    g.dec.linalg[1] = dest_stride;
    printf("  linear algebra: the %ux%u matrix as fp16 in the %s layout, %u bytes at stride %u, with the %s bias at "
           "offset %u; the buffer is %u bytes\n", (unsigned)LA_M, (unsigned)LA_K,
           g_la_mul_optimal ? "device's multiply-optimal" : "row-major", (unsigned)dest_size, (unsigned)dest_stride,
           g_la_fp32 ? "fp32" : "fp16", (unsigned)bias_offset, (unsigned)g_la_bytes);
    printf("  linear algebra: the largest this decoder's output can reach in fp16 is %g of 65504 (output row %d), so the "
           "range bound passes\n", (double)worst, worst_row);
    return true;
}
#endif

// ---------------------------------------------------------------------------
// Debug text overlay (8x8 font rasterised to an RGBA texture, drawn as a quad).
//
// The rasterisation is the Direct3D 11 viewer's, character for character, because nothing about a bitmap font is
// api-specific and that viewer's four lines have to be the same bytes here for the cross-viewer comparison to mean
// anything. This viewer draws a fifth line of its own below those four, which is additive and leaves their rows alone.
// What is new is the upload: the strip goes into a default-heap texture through an upload buffer at the top of the
// frame's command list, and only when the text has changed.
// ---------------------------------------------------------------------------
static const int OVL_W = 2560, FONT_SCALE = 2, LINE_ADV = 20;   // the overlay strip spans the wider window
// The strip's height, which is the number of lines it carries: the four the Direct3D 11 viewer draws too, and below
// them a fifth that says which decoder drew the frame. EVERY build draws that fifth line, whether or not it was made
// with the optional Shader Model 6.10 decode path, because which decoder drew a frame is a fact about the frame and
// not about a CMake option - a build with one path says it has one. The strip therefore grows downwards by one line
// and the help page below it follows on its own, since OVL_TEX_H is measured from here.
static const int OVL_H = 84 + LINE_ADV;
// The help page (key F1): every key the viewer handles, drawn below the status strip in the same texture. Any key closes it.
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
    "  M           mips on/off (off: every fetch reads mip 0)",
    "  L           level 1's LOD shift on/off (its gradients x 2^lod_bias_level1)",
    "  4           level 0 from the load-time BC4/BC5 pack on/off (uncompressed level 0 only)",
    "Other:",
    "  5 6 7 8     the shader's spare debug constants (unused by the shipped shader)",
#if NNTC_D3D12_LINALG
    "  K           the decode path: the Shader Model 6.10 one or the plain one (where the device offers it)",
#endif
    "  R           reload the shader (nntc_view.hlsl)",
    "  F1          this page",
    "  Esc         quit (with this page open: close the page only)",
};
static const int HELP_N = (int)(sizeof(HELP_LINES) / sizeof(HELP_LINES[0]));
static const int OVL_TEX_H = OVL_H + 8 + HELP_N * LINE_ADV;   // the texture holds the strip and the help page below it
static bool g_help = false;    // F1: the help page is shown
static int  g_help_eat = 0;    // the key that closed the help page: ignored by the held-key scan until it is released
static bool g_overlay = true;  // --nooverlay: draw the scene alone, so two --shot frames can be compared byte for byte

static void blit_char(std::vector<uint8_t>& buf, int px, int py, char ch) {
    int c = (unsigned char)ch; if (c<32 || c>127) c='.';
    const uint8_t* glyph = g_font8x8[c-32];
    for (int y=0;y<8;y++) for (int x=0;x<8;x++) {
        if (!((glyph[y]>>x)&1)) continue;
        for (int sy=0;sy<FONT_SCALE;sy++) for (int sx=0;sx<FONT_SCALE;sx++) {
            int X=px+x*FONT_SCALE+sx, Y=py+y*FONT_SCALE+sy;
            if (X<0||X>=OVL_W||Y<0||Y>=OVL_TEX_H) continue;
            uint8_t* d=&buf[((size_t)Y*OVL_W+X)*4]; d[0]=255;d[1]=255;d[2]=255;d[3]=255;
        }
    }
}
#if NNTC_D3D12_LINALG
// The width the DECODE LINE below has to live inside, and the reason string cut down to fit it. The strip is 2560
// pixels wide and is drawn one pixel to one pixel from the left edge of the frame, so a window narrower than that sees
// only its left-hand end: at 1280 pixels, the first eighty characters of a line and nothing beyond them. That is the
// budget, and it is the whole reason this line is on the left of the strip rather than at the right-hand end where the
// decode path used to be written.
//
// The refusals on this path are written as one clause and then an aside - an opening parenthesis, a "so", an "and the",
// a semicolon - so the aside is dropped and what is left is cut at a word boundary to whatever room the line has. The
// reason in full is on stdout at start-up, on the viewer's own `linear algebra: not available` line, which is where a
// reader who wants the rest of it looks.
static const size_t OVL_LINE_CHARS = 1280 / (8 * FONT_SCALE);   // what the narrowest window here shows of one line
static std::string short_why_not(size_t room) {
    std::string s = g_la_why_not.empty() ? "the device query did not pass" : g_la_why_not;
    auto trim = [](std::string& t) { while (!t.empty() && (t.back() == ',' || t.back() == ';' || t.back() == ' ')) t.pop_back(); };
    static const char* const asides[] = { " (", ", so ", "; ", " and the " };
    size_t cut = s.size();
    for (const char* a : asides) { const size_t at = s.find(a); if (at != std::string::npos) cut = std::min(cut, at); }
    s.resize(cut);
    trim(s);
    if (s.size() > room) {
        s.resize(room);
        const size_t space = s.rfind(' ');
        if (space != std::string::npos) s.resize(space);
        trim(s);
    }
    return s;
}
#endif

// The strip's five lines, rasterised into `buf`: the four the Direct3D 11 viewer draws too, and the decode line below
// them. The four are that viewer's text, word for word, because the gate asserts that the rows they occupy are the
// same bytes in the two viewers, over the whole width of the strip.
static void rasterise_overlay(std::vector<uint8_t>& buf) {
    buf.assign((size_t)OVL_W*OVL_TEX_H*4, 0);
    for (size_t i=0;i<(size_t)OVL_W*OVL_H*4;i+=4){ buf[i]=0;buf[i+1]=0;buf[i+2]=0;buf[i+3]=180; }
    // The help page's backdrop is only as wide as its longest line; below the strip the rest stays transparent.
    int help_w = 0; for (int i=0;i<HELP_N;i++) help_w = std::max(help_w, (int)strlen(HELP_LINES[i]));
    help_w = std::min(OVL_W, 8 + help_w*8*FONT_SCALE);
    for (int y=OVL_H;y<OVL_TEX_H;y++) for (int x=0;x<help_w;x++) buf[((size_t)y*OVL_W+x)*4+3]=200;
    char l0[256], l1[256], l2[256];
    // The load-time pack's own name: one BC4 or BC5, and for three or four channels the second texture's format too
    // (three channels put channel 2 alone in a BC4, four put channels 2-3 in a second BC5).
    const char* packname = g.lat[0].C == 1 ? "BC4" : (g.lat[0].C == 2 ? "BC5" : (g.lat[0].C == 3 ? "BC5+BC4" : "BC5+BC5"));
    char bcs[64]; if (g.lat[0].bc_n == 0) snprintf(bcs, sizeof(bcs), "U8"); else if (g.lat[0].bc_lossless) snprintf(bcs, sizeof(bcs), "%s lossless", packname); else snprintf(bcs, sizeof(bcs), "%s %.1fdB", packname, g.lat[0].bc_psnr);
    auto fmt_name = [](int dxgi) { return dxgi == 61 ? "R8" : (dxgi == 49 ? "R8G8" : (dxgi == 28 ? "RGBA8" : (dxgi == 80 ? "BC4" : (dxgi == 83 ? "BC5" : "?")))); };   // the texture's stored format
    // The format line when level 0 arrived block-compressed in the file: nothing was packed here, so it says "(file)" and no quality.
    char bcf[64]; snprintf(bcf, sizeof(bcf), "%s%s%s (file)", fmt_name(g.lat[0].dxgi), g.lat[0].file_files > 1 ? "+" : "", g.lat[0].file_files > 1 ? fmt_name(g.lat[0].dxgi_b) : "");
    // GPU memory of what is bound, in bytes, summed over the base and over the whole chain, as bits per source pixel of the decode size
    // per material texture. An uncompressed texture is one byte per stored channel per texel; a BLOCK-COMPRESSED one is whole 4x4
    // blocks - ceil(w/4) * ceil(h/4) of them at 8 bytes (BC4) or 16 (BC5) - which is what the driver allocates and what the .dds holds.
    double mem_base = 0, mem_all = 0, mem_base_l0 = 0;
    for (int l = 0; l < 2; l++) {
        const LatentTex& L = g.lat[l]; const bool packed = l == 0 && g_bc && L.bc_n > 0;
        // Each bound texture counted at ITS OWN rate, because the two textures of a three-channel level 0 are a BC5 and a
        // BC4 and not two of either: 16 + 8 bytes a block, which is the 12 bpp the file costs.
        double bytes_per_block = 0.0;   // zero when nothing here is block-compressed
        if (packed) { for (int p = 0; p < L.bc_n; p++) bytes_per_block += L.bc_nc[p] == 1 ? 8.0 : 16.0; }
        else if (L.file_bc) bytes_per_block = (L.dxgi == 80 ? 8.0 : 16.0) + (L.file_files > 1 ? (L.dxgi_b == 80 ? 8.0 : 16.0) : 0.0);
        for (int i = 0; i < (int)L.dims.size(); i++) {
            const int w = L.dims[(size_t)i].first, h = L.dims[(size_t)i].second;
            const double b = bytes_per_block > 0.0 ? (double)((w + 3) / 4) * ((h + 3) / 4) * bytes_per_block
                                                   : (double)L.stored_C * w * h;
            if (i == 0) { mem_base += b; if (l == 0) mem_base_l0 = b; }
            mem_all += b;
        }
    }
    const double px = (double)std::max(1, g.lat[0].W) * std::max(1, g.lat[0].H) * std::max(1, g.textures_out);   // the decode size x the textures
    // Level 0's own share is called out beside the total: it is the number the file layout decides (4 bpp for one BC4,
    // 8 for one BC5, 12 for a BC5 + a BC4 at three channels, 16 for two BC5s), divided by the material's textures.
    const double bpp_base = 8.0 * mem_base / px, bpp_all = 8.0 * mem_all / px, bpp_l0 = 8.0 * mem_base_l0 / px;
    snprintf(l0,sizeof(l0),"Src:%dx%d  L0:%dx%dx%d %s idx%db %dmips  L1:%dx%dx%d %s idx%db %dmips  blk:%d  GPU:%.2fbpp/tex base (L0 %.2f), %.2f with mips",
             g.src_w,g.src_h, g.lat[0].W,g.lat[0].H,g.lat[0].C, g.lat[0].file_bc ? bcf : (g_bc && g.lat[0].bc_n ? bcs : fmt_name(g.lat[0].dxgi)), g.lat[0].bits[0], g.lat[0].mips, g.lat[1].W,g.lat[1].H,g.lat[1].C, fmt_name(g.lat[1].dxgi), g.lat[1].bits[0], g.lat[1].mips, g.block, bpp_base, bpp_l0, bpp_all);
    // Anisotropy applies in trilinear mode only, so in the other two the overlay says the flag is remembered and idle.
    char aniso_s[32];
    if (!g.aniso) snprintf(aniso_s, sizeof(aniso_s), "Aniso:OFF");
    else if (g.filter_mode == 2) snprintf(aniso_s, sizeof(aniso_s), "Aniso:%u", (unsigned)ANISO_MAX);
    else snprintf(aniso_s, sizeof(aniso_s), "Aniso:%u (trilinear only)", (unsigned)ANISO_MAX);
    snprintf(l1,sizeof(l1),"Mode:%-4s Filter:%-9s %-24s Mips:%s L1lod:%s Tex:%d/%d  Show:%s  Renorm:%s",
             g.cube?"CUBE":"QUAD", filter_name(g.filter_mode), aniso_s, g.mips_on?"ON ":"OFF", g.lod_bias?"ON ":"OFF", g.tex_shown, g.textures_out,
             g.const0[0]>0.5f?"LATENT0":(g.const0[1]>0.5f?"LATENT1":"DECODE"), g.const0[3]>0.5f?"ON ":"OFF");
    snprintf(l2,sizeof(l2),"X:%+5.1f Y:%+5.1f Z:%5.1f Yaw:%+6.1f Pitch:%+6.1f", g.x,g.y,g.z,g.yaw,g.pitch);
    const char* l3 = "F1=Help  Move:Arrows/WS Rot:ADQE C:cube B/T/P:filter X:aniso M:mips L:L1lod N:tex V:renorm 1/2:latents 4:BC-pack R:reload Spc:reset Esc";
    const char* lines[4] = {l0,l1,l2,l3};
    for (int li=0; li<4; li++) { int y = 2 + li*LINE_ADV, x = 4; for (const char* p=lines[li]; *p; ++p) { blit_char(buf, x, y, *p); x += 8*FONT_SCALE; } }
    for (int li=0; li<HELP_N; li++) { int y = OVL_H + 6 + li*LINE_ADV, x = 4; for (const char* p=HELP_LINES[li]; *p; ++p) { blit_char(buf, x, y, *p); x += 8*FONT_SCALE; } }
    // THE DECODE LINE, the strip's fifth, in the same font, the same scale and at the same left margin as the four
    // above it, and drawn by EVERY build. It says which decoder drew the frame in words rather than in a token, and it
    // says it on the left: this used to be a ten-character token at the right-hand end of a 2560-pixel strip, which on
    // a 1920- or a 1280-wide window was off the side of the frame entirely and was never once seen on a monitor.
    //
    // Four states and not two. The plain path drawing on a machine that offers the other one (--linalg 0, or the key K)
    // is a CHOICE and reads as one; the plain path drawing because this DEVICE has no other is a fact about the run and
    // carries the reason, so that a reader does not have to go back to the start-up output to learn it; and the plain
    // path drawing because this BUILD has no other is a fact about the build, which is the one case that needs no
    // reason - there is nothing here to have gone wrong.
    char l4[256];
#if NNTC_D3D12_LINALG
    if (g_decode == DECODE_LINALG)
        snprintf(l4, sizeof(l4), "Decode: linear algebra, through Shader Model 6.10");
    else if (g_la_available)
        snprintf(l4, sizeof(l4), "Decode: plain, though this device offers linear algebra");
    else
        snprintf(l4, sizeof(l4), "Decode: plain, no linear algebra - %s",
                 short_why_not(OVL_LINE_CHARS - strlen("Decode: plain, no linear algebra - ")).c_str());
#else
    snprintf(l4, sizeof(l4), "Decode: plain, no linear algebra in this build");
#endif
    // Four rows into its own line rather than two, which is where the other four sit: a glyph is 16 rows tall and a
    // line advances 20, so a fifth line begun at the usual offset would put its first two rows inside the 84-row band
    // the other two viewers' strips also fill, and the comparison against them would have to give up those two rows.
    // The strip has room for it - the line's glyphs end at 99 of 104 - and the band below 84 is this viewer's alone.
    { int y = 4 + 4 * LINE_ADV, x = 4; for (const char* p = l4; *p; ++p) { blit_char(buf, x, y, *p); x += 8 * FONT_SCALE; } }
}
static bool init_overlay() {
    const D3D12_RESOURCE_DESC rd = texture_desc(DXGI_FORMAT_R8G8B8A8_UNORM, OVL_W, OVL_TEX_H, 1);
    const D3D12_HEAP_PROPERTIES def = heap_props(D3D12_HEAP_TYPE_DEFAULT);
    if (FAILED(g_dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&g.debug_tex)))) return false;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{}; UINT rows = 0; UINT64 row_bytes = 0, total = 0;
    g_dev->GetCopyableFootprints(&rd, 0, 1, 0, &fp, &rows, &row_bytes, &total);
    g.debug_upload = make_upload_buffer(nullptr, (size_t)total);
    if (!g.debug_upload) return false;
    g.debug_vb = make_upload_buffer(nullptr, 16 * sizeof(float));
    const uint32_t idx[] = {0,1,2, 0,2,3};
    g.debug_ib = make_upload_buffer(idx, sizeof(idx));
    if (!g.debug_vb || !g.debug_ib) return false;
    write_srv(SRV_OVERLAY + 0, g.debug_tex, DXGI_FORMAT_R8G8B8A8_UNORM, 1);
    write_srv(SRV_OVERLAY + 1, nullptr, DXGI_FORMAT_R8G8B8A8_UNORM, 1);
    write_srv(SRV_OVERLAY + 2, nullptr, DXGI_FORMAT_R8G8B8A8_UNORM, 1);
    return true;
}
// Recorded at the top of the frame's command list, before the render target is set, because a copy into a texture the
// pixel shader is about to read has to be a copy and not a write under the draw.
static void upload_overlay_if_dirty() {
    if (!g.debug_dirty || !g.debug_tex) return;
    std::vector<uint8_t> buf;
    rasterise_overlay(buf);
    const D3D12_RESOURCE_DESC rd = g.debug_tex->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{}; UINT rows = 0; UINT64 row_bytes = 0, total = 0;
    g_dev->GetCopyableFootprints(&rd, 0, 1, 0, &fp, &rows, &row_bytes, &total);
    uint8_t* p = nullptr; D3D12_RANGE none{ 0, 0 };
    if (FAILED(g.debug_upload->Map(0, &none, (void**)&p))) return;
    for (UINT y = 0; y < rows; y++) memcpy(p + fp.Offset + (SIZE_T)y * fp.Footprint.RowPitch, &buf[(size_t)y * OVL_W * 4], (size_t)row_bytes);
    g.debug_upload->Unmap(0, nullptr);
    if (g.debug_in_ps) barrier(g.debug_tex, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource = g.debug_tex; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dst.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource = g.debug_upload; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = fp;
    g_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    barrier(g.debug_tex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    g.debug_in_ps = true;
    g.debug_dirty = false;
}
static void draw_overlay() {
    if (!g.pso_overlay || !g_overlay || !g.debug_in_ps) return;
    const int rows = g_help ? OVL_TEX_H : OVL_H;   // the strip alone, or the strip and the help page below it
    // The Direct3D 11 viewer's own arithmetic: the strip is OVL_W pixels wide at 1:1 and is scaled by the frame's width,
    // so at the 2560-wide frame both viewers put the same pixels in the same place, which is what the gate compares.
    float w = (float)OVL_W / g_frame_w * 2.0f, h = (float)rows / g_frame_h * 2.0f, v = (float)rows / OVL_TEX_H;
    float verts[] = { -1.0f, 1.0f, 0.0f, 0.0f,  -1.0f + w, 1.0f, 1.0f, 0.0f,  -1.0f + w, 1.0f - h, 1.0f, v,  -1.0f, 1.0f - h, 0.0f, v };
    uint8_t* p = nullptr; D3D12_RANGE none{ 0, 0 };
    if (SUCCEEDED(g.debug_vb->Map(0, &none, (void**)&p))) { memcpy(p, verts, sizeof(verts)); g.debug_vb->Unmap(0, nullptr); }
    D3D12_VERTEX_BUFFER_VIEW vbv{}; vbv.BufferLocation = g.debug_vb->GetGPUVirtualAddress(); vbv.SizeInBytes = (UINT)sizeof(verts); vbv.StrideInBytes = 4 * sizeof(float);
    D3D12_INDEX_BUFFER_VIEW ibv{}; ibv.BufferLocation = g.debug_ib->GetGPUVirtualAddress(); ibv.SizeInBytes = 6 * sizeof(uint32_t); ibv.Format = DXGI_FORMAT_R32_UINT;
    g_list->SetPipelineState(g.pso_overlay);
    g_list->SetGraphicsRootDescriptorTable(2, gpu_handle(g_srv_heap, g_srv_step, SRV_OVERLAY));
    g_list->SetGraphicsRootDescriptorTable(3, gpu_handle(g_samp_heap, g_samp_step, 0));   // the point sampler with MaxLOD 0, as the other viewer binds for the strip
    g_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    g_list->IASetVertexBuffers(0, 1, &vbv);
    g_list->IASetIndexBuffer(&ibv);
    g_list->DrawIndexedInstanced(6, 1, 0, 0, 0);
}

// ---------------------------------------------------------------------------
// Input
//
// --shot TAKES NO INPUT AT ALL, and in this viewer it also has no window to receive any: the headless path creates no
// window class, no window and no swap chain. The rule is kept here as well, because the same wnd_proc and the same
// held-key scan serve both paths and a shot that read the keyboard would not be a function of its command line.
// ---------------------------------------------------------------------------
static bool g_shot = false;   // --shot: one frame, no window, no input
static void resize_swapchain(int w, int h);
static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_CLOSE: case WM_DESTROY: g_quit = true; PostQuitMessage(0); return 0;
        case WM_SIZE: if (wp != SIZE_MINIMIZED) resize_swapchain((int)LOWORD(lp), (int)HIWORD(lp)); return 0;
        case WM_SYSKEYDOWN:   // Alt and F10 arrive here: they close the help page too, and otherwise go to DefWindowProc as before
            if (!g_shot && g_help && !(lp & (1 << 30))) { g_help = false; g_help_eat = (int)wp; g.debug_dirty = true; return 0; }
            break;
        case WM_KEYDOWN: {
            if (g_shot) return 0;   // a keystroke that lands in the shot window changes nothing it draws
            if (lp & (1 << 30)) return 0;
            // With the help page open any key closes it and does nothing else: Esc closes the page, not the viewer, and a
            // held movement key that closed it does not move the camera until it is released.
            if (g_help) { g_help = false; g_help_eat = (int)wp; g.debug_dirty = true; return 0; }
            switch (wp) {
                case VK_F1: g_help = true; g.debug_dirty = true; break;
                case VK_ESCAPE: g_quit = true; break;
                case 'R': reload_shader(); break;
#if NNTC_D3D12_LINALG
                case 'K':
                    if (!g_la_available) { printf("decode: plain (%s)\n", g_la_why_not.c_str()); break; }
                    if (!g.pso_linalg) { printf("decode: plain (--linalg 0 was given, so the Shader Model 6.10 pipeline was never built)\n"); break; }
                    g_decode = g_decode == DECODE_LINALG ? DECODE_PLAIN : DECODE_LINALG;
                    g.debug_dirty = true;
                    printf("decode: %s\n", decode_path_name());
                    break;
#endif
                case 'B': g.filter_mode=1; g.debug_dirty=true; printf("Filter: BILINEAR\n"); break;
                case 'T': g.filter_mode=2; g.debug_dirty=true; printf("Filter: TRILINEAR\n"); break;
                case 'P': g.filter_mode=0; g.debug_dirty=true; printf("Filter: POINT\n"); break;
                case 'X': g.aniso=!g.aniso; g.debug_dirty=true; printf("Anisotropic filtering: %s%s\n", g.aniso ? "ON (MaxAnisotropy " : "OFF", g.aniso ? (std::to_string((unsigned)ANISO_MAX) + ")").c_str() : ""); if (g.filter_mode != 2) printf("  (it applies in TRILINEAR mode only; press T)\n"); break;
                case 'L': g.lod_bias=!g.lod_bias; g.debug_dirty=true; printf("Level 1 LOD shift: %s\n", g.lod_bias ? "ON (gradients scaled by 2^lod_bias_level1: the 1:1 mip rule)" : "OFF (the GPU's own LOD: level 1 two mips finer than fitted)"); break;
                case 'M': g.mips_on=!g.mips_on; g.debug_dirty=true; printf("Mips: %s\n", g.mips_on ? "ON (MaxLOD unlimited)" : "OFF (sampler MaxLOD = 0: mip 0 only)"); break;
                case 'C': g.cube=!g.cube; g.debug_dirty=true; break;
                case 'N': g.tex_shown = (g.tex_shown + 1) % g.textures_out; g.debug_dirty=true; printf("Output texture %d of %d\n", g.tex_shown, g.textures_out); break;
                case '1': g.const0[0]=1.0f-g.const0[0]; g.const0[1]=0; g.debug_dirty=true; break;
                case '2': g.const0[1]=1.0f-g.const0[1]; g.const0[0]=0; g.debug_dirty=true; break;
                case 'V': g.const0[3]=1.0f-g.const0[3]; g.debug_dirty=true; printf("Normal renormalisation: %s\n", g.const0[3]>0.5f ? "ON (unpack, unit length, repack)" : "OFF"); break;
                case '4': if (g.lat[0].bc_n) { g_bc = !g_bc; g.debug_dirty=true; printf("Level 0: %s\n", g_bc ? "BC4/BC5 pack" : "uncompressed"); } break;
                case '5': g.const1[0]=1.0f-g.const1[0]; g.debug_dirty=true; break;
                case '6': g.const1[1]=1.0f-g.const1[1]; g.debug_dirty=true; break;
                case '7': g.const1[2]=1.0f-g.const1[2]; g.debug_dirty=true; break;
                case '8': g.const1[3]=1.0f-g.const1[3]; g.debug_dirty=true; break;
                case VK_SPACE:
                    g.x=0;g.y=0;g.z=-3.0f;g.yaw=0;g.pitch=0;
                    for(int i=0;i<4;i++){g.const0[i]=0;g.const1[i]=0;}
                    g.tex_shown = 0; g.debug_dirty=true; printf("Reset to initial state\n"); break;
            }
            return 0;
        }
        default: break;
    }
    return DefWindowProcA(hwnd, msg, wp, lp);
}
static bool key_down(int vk) { return vk != g_help_eat && (GetAsyncKeyState(vk) & 0x8000) != 0; }
static void process_held_keys(HWND hwnd, float dt) {
    if (g_shot) return;   // see above: the shot frame is a function of the command line and the asset, and of nothing else
    if (GetForegroundWindow() != hwnd) return;
    if (g_help) return;   // the help page is modal: nothing moves while it is open
    if (g_help_eat && !(GetAsyncKeyState(g_help_eat) & 0x8000)) g_help_eat = 0;   // the key that closed the page is up again
    if (key_down(VK_SHIFT)) dt *= 1.0f/3.0f;
    bool moved=false;
    if (key_down('W')) {g.z+=Z_SPEED*dt;moved=true;}
    if (key_down('S')) {g.z-=Z_SPEED*dt;moved=true;}
    if (key_down(VK_LEFT))  {g.x+=XY_SPEED*dt;moved=true;}
    if (key_down(VK_RIGHT)) {g.x-=XY_SPEED*dt;moved=true;}
    if (key_down(VK_UP))    {g.y+=XY_SPEED*dt;moved=true;}
    if (key_down(VK_DOWN))  {g.y-=XY_SPEED*dt;moved=true;}
    if (key_down('A')) {g.yaw+=ROT_SPEED*dt;moved=true;}
    if (key_down('D')) {g.yaw-=ROT_SPEED*dt;moved=true;}
    if (key_down('Q')) {g.pitch+=ROT_SPEED*dt;moved=true;}
    if (key_down('E')) {g.pitch-=ROT_SPEED*dt;moved=true;}
    if (g.z < Z_MAX) g.z=Z_MAX; if (g.z > Z_MIN) g.z=Z_MIN;
    if (moved) g.debug_dirty=true;
}

// ---------------------------------------------------------------------------
// The constant buffers, written into the two upload-heap allocations that are mapped for the program's whole life.
// ---------------------------------------------------------------------------
struct CBData {          // must match cbuffer SceneConstants in nntc_view.hlsl (16-byte rules)
    float mvp[16];
    float texSize[4];
    float lodInfo[4];
    float const0[4];
    float const1[4];
};
static void set_uniforms() {
    Mat4 proj = mat_perspective(FOV_DEGREES, (float)g_frame_w/g_frame_h, 0.001f, 100.0f);
    Mat4 model = mat_mul(mat_mul(mat_translate(g.x,g.y,g.z), mat_rot_y(g.yaw)), mat_rot_x(g.pitch));
    Mat4 mvp = mat_mul(proj, model);
    CBData cb{};
    Mat4 mvp_t = mat_transpose(mvp);
    memcpy(cb.mvp, mvp_t.m, sizeof(cb.mvp));
    cb.texSize[0]=(float)g.lat[0].W; cb.texSize[1]=(float)g.lat[0].H; cb.texSize[2]=(float)g.lat[1].W; cb.texSize[3]=(float)g.lat[1].H;
    cb.lodInfo[0]=(float)(g.lat[0].mips-1); cb.lodInfo[1]=(float)(g.lat[1].mips-1);
    memcpy(cb.const0, g.const0, sizeof(cb.const0));
    memcpy(cb.const1, g.const1, sizeof(cb.const1));
    // const0.z: the factor the shader multiplies level 1's UV derivatives by before its SampleGrad. 2^lod_bias_level1
    // raises the hardware's LOD by lod_bias_level1 mips, which is the encoder's 1:1 rule; key L off leaves the
    // gradients alone, which is the control. It is written here every frame rather than kept in g.const0, so no key
    // and no reset can leave a zero scale in the slot.
    cb.const0[2] = g.lod_bias ? std::exp2(g.lod_bias_level1) : 1.0f;
    memcpy(g_cb_scene_ptr, &cb, sizeof(cb));
    g.dec.sel[1] = g.tex_shown;
    memcpy(g_cb_dec_ptr, &g.dec, sizeof(g.dec));
}

// EVERY flag, not two of them; the Direct3D 11 viewer's spellings, plus the Vulkan viewer's --size and --device.
static void viewer_usage(const char* exe) {
    printf("Usage: %s <PREFIX_nntc.json> [flags]\n", exe);
    printf("  The asset's descriptor and the .dds files it names, as written by nntc_encode, drawn through a Direct3D 12\n");
    printf("  sampler. The path is taken as given: the viewer derives no name of its own.\n\n");
    printf("  --shot FILE.bmp  render ONE frame to a 24-bit .bmp and exit, with NO WINDOW and no swap chain at all, so\n");
    printf("                   the bytes are a function of the command line and the asset alone (a check without a desktop)\n");
    printf("  --tex N          show output texture N of a material (the key N cycles them)\n");
    printf("  --cube           the cube rather than the quad (the key C)\n");
    printf("  --z F            the camera distance\n");
    printf("  --yaw F          the camera yaw, in degrees\n");
    printf("  --pitch F        the camera pitch, in degrees\n");
    printf("  --nomips         the sampler's MaxLOD is 0, so every fetch reads mip 0 (the key M off)\n");
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
    printf("  --size W H       the --shot frame's size, 2560x1440 by default, and the window's, 1280x720 by default\n");
    printf("                   (drag it to change it; the picture follows). The frame default exists for the comparison\n");
    printf("                   against the Direct3D 11 viewer, whose --shot is its WINDOW's client area and is therefore\n");
    printf("                   the desktop's work area wherever that is smaller; two frames of different shapes are two\n");
    printf("                   different projections and cannot be compared\n");
    printf("  --device N       draw on DXGI adapter N instead of the first one with a Direct3D 12 device (the adapters\n");
    printf("                   are listed at start-up, WARP last, and the chosen one is named either way)\n");
    printf("  --linalg 0|1     the decode through the Shader Model 6.10 linear-algebra instruction (1) or through the\n");
    printf("                   plain matrix multiply (0); the key K switches. It is ON BY DEFAULT WHERE THE DEVICE AND\n");
    printf("                   THE BUILD OFFER IT, and --linalg 1 where either does not is refused - a run that\n");
    printf("                   quietly fell back would be a measurement of the other path under this one's name\n");
    printf("  --linalg-layout row|optimal\n");
    printf("                   which layout the fp16 matrix is read in: row-major as the host writes it, or the\n");
    printf("                   device's own multiply-optimal layout through ConvertLinearAlgebraMatrix. The default is\n");
    printf("                   the optimal one where the runtime offers it, and both are meant to draw one picture\n");
    printf("  --bench N        render N frames headless, with no window and no file written, and print the mean and the\n");
    printf("                   minimum GPU time of the scene draw in microseconds. INFORMATIONAL ONLY: this viewer is a\n");
    printf("                   correctness sample and speed is not one of its claims. It times the CURRENT decode path\n");
    printf("  --help, -h       this text\n\n");
    printf("  The shader is nntc_view.hlsl, read from beside the executable and from nowhere else and compiled at\n");
    printf("  runtime; the key R re-reads it from there. It is the Direct3D 11 viewer's own file, unchanged.\n");
}

// A numeric flag's value must be the whole argument and finite: `--tex nope` read through atoi is texture 0, which is the
// kind of silent acceptance neither the encoder nor the other viewers allow.
static bool parse_number(const char* text, double& out) {
    char* end = nullptr; errno = 0; const double v = strtod(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE || !std::isfinite(v)) return false;
    out = v; return true;
}
static bool parse_integer(const char* text, int& out) {
    char* end = nullptr; errno = 0; const long v = strtol(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE || v < -1000000 || v > 1000000) return false;
    out = (int)v; return true;
}

// ---------------------------------------------------------------------------
// THE LINEAR-ALGEBRA QUERY, in the Vulkan viewer's order and with the Direct3D 12 spellings (the plan's B.3). Every step
// prints what it found, and a step that fails prints exactly one line, `linear algebra: not available - <why>`, after
// which the plain path draws. NONE OF IT IS A FAILURE OF THE PROGRAM: only --linalg 1 turns a failed step into an exit,
// and it does that at one place, in init_common.
//
// The diagnostics are deliberately verbose, because the machine this path has to work on is not this one: a run's own
// output has to be enough to say which step stopped it (viewer_d3d12/PHASE_B_NOTES.md reads these lines one by one).
// ---------------------------------------------------------------------------
#if NNTC_D3D12_LINALG
static const char* la_datatype_name(D3D12_LINEAR_ALGEBRA_DATATYPE t) {
    switch (t) {
        case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16: return "fp16";
        case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32: return "fp32";
        case D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8: return "sint8";
        case D3D12_LINEAR_ALGEBRA_DATATYPE_UINT8: return "uint8";
        case D3D12_LINEAR_ALGEBRA_DATATYPE_SINT16: return "sint16";
        case D3D12_LINEAR_ALGEBRA_DATATYPE_UINT16: return "uint16";
        case D3D12_LINEAR_ALGEBRA_DATATYPE_SINT32: return "sint32";
        case D3D12_LINEAR_ALGEBRA_DATATYPE_UINT32: return "uint32";
        case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT8_E4M3FN: return "e4m3";
        case D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT8_E5M2: return "e5m2";
        default: return "?";
    }
}
// The flags in words, EMULATED_OUTPUTS included: a Required row may still report it, and an fp16 result type is not an
// fp16 accumulation - the implementation is free to accumulate in fp32 and round at the end, and this flag is precisely
// how it says it converted the result. So the flags of the row that actually ran are printed and nothing about the
// arithmetic is asserted (the plan's B.1.1, "wording (review)").
static std::string la_flags_text(D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAGS f) {
    std::string s;
    if (f & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_SUPPORTED) s += "SUPPORTED";
    if (f & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_EMULATED_INPUTS) s += s.empty() ? "EMULATED_INPUTS" : " | EMULATED_INPUTS";
    if (f & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_EMULATED_OUTPUTS) s += s.empty() ? "EMULATED_OUTPUTS" : " | EMULATED_OUTPUTS";
    if (f & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_TRANSPOSE) s += s.empty() ? "TRANSPOSE" : " | TRANSPOSE";
    if (s.empty()) s = "none";
    return s;
}
// WHICH D3D12Core.dll this process actually loaded, and its version. This is the silent-fallback diagnostic: when the
// preview redistributable was not picked up - a missing D3D12\ folder, a wrong D3D12SDKPath, an export the linker
// dropped - the system runtime answers every query below with a polite "no" and the run would otherwise look like a
// device that simply lacks the feature. The path says which it was.
static void print_loaded_module(const wchar_t* name) {
    const HMODULE m = GetModuleHandleW(name);
    if (!m) { printf("  %ls: not loaded by this process\n", name); return; }
    wchar_t path[MAX_PATH * 2] = { 0 };
    if (!GetModuleFileNameW(m, path, (DWORD)(sizeof(path) / sizeof(path[0])))) { printf("  %ls: loaded, path unknown\n", name); return; }
    std::string version = "no version resource";
    DWORD ignored = 0;
    const DWORD size = GetFileVersionInfoSizeW(path, &ignored);
    if (size) {
        std::vector<uint8_t> buf((size_t)size);
        VS_FIXEDFILEINFO* fi = nullptr; UINT n = 0;
        if (GetFileVersionInfoW(path, 0, size, buf.data()) && VerQueryValueW(buf.data(), L"\\", (void**)&fi, &n) && fi) {
            char v[64];
            snprintf(v, sizeof(v), "%u.%u.%u.%u", (unsigned)HIWORD(fi->dwFileVersionMS), (unsigned)LOWORD(fi->dwFileVersionMS),
                     (unsigned)HIWORD(fi->dwFileVersionLS), (unsigned)LOWORD(fi->dwFileVersionLS));
            version = v;
        }
    }
    printf("  %ls: %s (%s)\n", name, narrow(path).c_str(), version.c_str());
}

static HRESULT g_la_experimental_hr = E_FAIL;
// Called ONCE, before any device exists, which is what the preview requires. The learn page warns that calling it a
// second time with a different list puts every device into DEVICE_REMOVED, so it is called from exactly one place.
static void enable_experimental_shader_models() {
    g_la_experimental_hr = D3D12EnableExperimentalFeatures(1, &D3D12ExperimentalShaderModels, nullptr, nullptr);
}

static void query_linear_algebra() {
    g_la_available = false;
    printf("linear algebra: the Shader Model 6.10 query, step by step\n");
    print_loaded_module(L"D3D12Core.dll");
    print_loaded_module(L"d3d10warp.dll");
    if (FAILED(g_la_experimental_hr)) {
        char why[288];
        snprintf(why, sizeof(why), "D3D12EnableExperimentalFeatures(D3D12ExperimentalShaderModels) returned 0x%08X%s",
                 (unsigned)g_la_experimental_hr,
                 g_la_experimental_hr == E_NOINTERFACE
                     ? " (E_NOINTERFACE: either Developer Mode is off, or the runtime that answered is the retail one, "
                       "which refuses experimental shader models by design)" : "");
        g_la_why_not = why;
        printf("linear algebra: not available - %s\n", g_la_why_not.c_str());
        return;
    }
    printf("  experimental shader models: enabled before device creation\n");

    // HighestShaderModel is the REQUEST going in and the answer coming back, so a call that failed leaves 6.10 sitting
    // in it - which is what the in-box runtime does with a shader model it has never heard of, the common case where
    // the preview files are missing. Printing that as the device's answer reads "highest answered 6.10" and then
    // refuses for want of 6.10, which is nonsense on its face, so a failed query is reported as a failed query.
    D3D12_FEATURE_DATA_SHADER_MODEL sm{ D3D_SHADER_MODEL_6_10 };
    HRESULT hr = g_dev->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm));
    if (FAILED(hr)) {
        char why[224];
        snprintf(why, sizeof(why), "the shader model query itself failed (0x%08X), so this runtime does not know "
                 "Shader Model 6.10", (unsigned)hr);
        g_la_why_not = why;
        printf("  shader model: 6.10 asked for, the query failed (0x%08X)\n", (unsigned)hr);
        printf("linear algebra: not available - %s\n", g_la_why_not.c_str());
        return;
    }
    printf("  shader model: 6.10 asked for, highest answered 0x%02X (%u.%u)\n", (unsigned)sm.HighestShaderModel,
           (unsigned)(sm.HighestShaderModel >> 4), (unsigned)(sm.HighestShaderModel & 0xF));
    if (sm.HighestShaderModel < D3D_SHADER_MODEL_6_10) {
        char why[224];
        snprintf(why, sizeof(why), "this device's highest shader model is %u.%u and the decode needs 6.10",
                 (unsigned)(sm.HighestShaderModel >> 4), (unsigned)(sm.HighestShaderModel & 0xF));
        g_la_why_not = why;
        printf("linear algebra: not available - %s\n", g_la_why_not.c_str());
        return;
    }

    // Same care as above: the struct is zero-initialised, so a query that failed leaves a tier of 0x0 behind, which
    // reads exactly like a device that answered NOT_SUPPORTED. The two are different facts and are said differently.
    D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT tier{};
    hr = g_dev->CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT, &tier, sizeof(tier));
    if (FAILED(hr)) {
        char why[224];
        snprintf(why, sizeof(why), "the linear algebra tier query itself failed (0x%08X), so this runtime does not "
                 "know the feature", (unsigned)hr);
        g_la_why_not = why;
        printf("  linear algebra tier: the query failed (0x%08X)\n", (unsigned)hr);
        printf("linear algebra: not available - %s\n", g_la_why_not.c_str());
        return;
    }
    printf("  linear algebra tier: 0x%X (%s)\n", (unsigned)tier.LinearAlgebraTier,
           tier.LinearAlgebraTier >= D3D12_LINEAR_ALGEBRA_TIER_1_0 ? "TIER_1_0" : "NOT_SUPPORTED");
    if (tier.LinearAlgebraTier < D3D12_LINEAR_ALGEBRA_TIER_1_0) {
        char why[224];
        snprintf(why, sizeof(why), "this device reports linear algebra tier 0x%X and the decode needs TIER_1_0 (0x10)",
                 (unsigned)tier.LinearAlgebraTier);
        g_la_why_not = why;
        printf("linear algebra: not available - %s\n", g_la_why_not.c_str());
        return;
    }

    // The granular rows. The mandatory all-fp16 one is asked for FIRST, because Tier 1 requires it and a device that
    // cannot do it is one this viewer has misread; then the fp32-bias-and-result row is asked for and taken when it is
    // granted, because an fp32 accumulation and an fp16 one are different numbers and which one ran has to be said.
    // The vector is fp16 under both: phi is rounded on the way in either way, which is what the shader does.
    //
    // (The header of this Agility SDK spells this query with the word repeated -
    // D3D12_FEATURE_LINEAR_ALGEBRA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT - where the runtime specification writes it
    // once. The header decides, and that is the plan's open question B.1.4 answered.)
    struct Row { D3D12_LINEAR_ALGEBRA_DATATYPE bias, result; bool fp32; };
    const Row rows[2] = {
        { D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16, D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16, false },
        { D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32, D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32, true },
    };
    bool have[2] = { false, false };
    for (int i = 0; i < 2; i++) {
        D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT op{};
        op.OperationType = D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_THREAD_VECTOR_MATRIX_MULTIPLY;
        op.ThreadVectorMatrixMultiply.VectorInputType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
        op.ThreadVectorMatrixMultiply.MatrixInputType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
        op.ThreadVectorMatrixMultiply.BiasInputType = rows[i].bias;
        op.ThreadVectorMatrixMultiply.VectorResultType = rows[i].result;
        const HRESULT rhr = g_dev->CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT, &op, sizeof(op));
        const D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAGS f = op.ThreadVectorMatrixMultiply.SupportFlags;
        printf("  row: vector fp16 x matrix fp16 + bias %s -> %s: %s%s\n", la_datatype_name(rows[i].bias),
               la_datatype_name(rows[i].result), FAILED(rhr) ? "the query itself failed, " : "", la_flags_text(f).c_str());
        have[i] = SUCCEEDED(rhr) && (f & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_SUPPORTED) != 0;
    }
    if (!have[0]) {
        g_la_why_not = "this device does not report the fp16 vector-matrix row that Tier 1 makes mandatory "
                       "(vector fp16 x matrix fp16 + bias fp16 -> fp16), so the decode has no row to run on";
        printf("linear algebra: not available - %s\n", g_la_why_not.c_str());
        return;
    }
    g_la_fp32 = have[1];

    // The enumeration query of the specification's 0.9 draft, which would print the driver's whole list. It is not in
    // this Agility SDK's header at all, and the draft says a v1 driver would not answer it either, so its absence is a
    // diagnostic and never a refusal.
    printf("  enumeration: not asked for - this Agility SDK's d3d12.h has no "
           "D3D12_FEATURE_LINEAR_ALGEBRA_OPERATION_ENUMERATION, and a v1 driver would not answer it\n");

    // The layout. The multiply-optimal one is the device's own and can only come from ConvertLinearAlgebraMatrix, so the
    // two Preview interfaces have to be there; where they are not, the row-major layout is used, which a thread-scope
    // load also permits and which needs no conversion at all. --linalg-layout pins either one for the comparison.
    ID3D12DevicePreview* devp = nullptr;
    ID3D12GraphicsCommandListPreview* listp = nullptr;
    const bool preview_ifaces = SUCCEEDED(g_dev->QueryInterface(IID_PPV_ARGS(&devp))) &&
                                g_list && SUCCEEDED(g_list->QueryInterface(IID_PPV_ARGS(&listp)));
    safe_release(devp); safe_release(listp);
    g_la_mul_optimal = preview_ifaces && g_la_layout_want != 0;
    if (g_la_layout_want == 1 && !preview_ifaces)
        fprintf(stderr, "WARNING: --linalg-layout optimal was asked for and the Preview interfaces are not available; "
                        "the row-major layout is used\n");
    printf("  matrix layout: %s (ID3D12DevicePreview and ID3D12GraphicsCommandListPreview %s)\n",
           g_la_mul_optimal ? "the device's multiply-optimal, through ConvertLinearAlgebraMatrix" : "row-major, no conversion",
           preview_ifaces ? "are both present" : "are not both present");

    if (!load_dxc()) {
        g_la_why_not = "dxcompiler.dll is not beside this executable, and the Shader Model 6.10 shader is compiled at "
                       "run time; the build copies it there, so an executable moved on its own loses the path";
        printf("linear algebra: not available - %s\n", g_la_why_not.c_str());
        return;
    }

    g_la_available = true;
    g_la_why_not.clear();
    printf("linear algebra: available - tier 1.0, Shader Model 6.10, vector fp16 x matrix fp16 + bias %s -> %s\n",
           g_la_fp32 ? "fp32" : "fp16", g_la_fp32 ? "fp32" : "fp16");
}
#else
// The same function in a build made without the option. It takes no argument because there is nothing to ask: the
// refusal is a property of the BUILD and not of the device, and it is printed in the same one line and the same shape,
// so a run's own output still says which decode path it could have taken.
static void query_linear_algebra() {
    g_la_available = false;
    g_la_why_not = "this build was made without the CMake option NNTC_D3D12_LINALG, so it carries no Shader Model 6.10 "
                   "shader, no Agility SDK and no preview runtime; the plain path is the product and runs everywhere";
    printf("linear algebra: not available - %s\n", g_la_why_not.c_str());
}
#endif

// ---------------------------------------------------------------------------
// The device. The adapters are enumerated and printed, WARP appended as the last entry, and --device N overrides the
// choice outright - which is how the same build is exercised on this machine's discrete GPU, its integrated part and the
// software rasteriser without a second binary. An adapter that cannot make a Direct3D 12 device at feature level 11_0 is
// listed with the reason after " - " and is a refusal rather than a second measurement, exactly as the Vulkan viewer's
// unusable devices are.
//
// "no Direct3D 12 device" is the phrase the release gate reads to SKIP its Direct3D 12 arms rather than fail them, so it
// belongs to the case where this machine cannot run the viewer at all and to nothing else.
// ---------------------------------------------------------------------------
static bool create_device(int want) {
    UINT factory_flags = 0;
#if NNTC_D3D12_LINALG
    // Before the debug layer, before the factory and before any device: the preview requires it, and calling it twice
    // with different lists removes every device. Its result is not acted on here - a failure is the query's first line
    // and the plain path still draws - so a machine without Developer Mode runs this viewer exactly as before.
    enable_experimental_shader_models();
#endif
#if ENABLE_D3D_DEBUG
    // The debug layer, GPU-based validation and DRED, all obtained before the device exists. A machine without the
    // Graphics Tools feature has no D3D12SDKLayers.dll; that is a WARNING and the viewer runs without them, as the
    // Direct3D 11 viewer's retry without its own debug layer does.
    {
        ID3D12Debug* dbg = nullptr;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) {
            dbg->EnableDebugLayer();
            factory_flags |= DXGI_CREATE_FACTORY_DEBUG;
            ID3D12Debug1* dbg1 = nullptr;
            if (SUCCEEDED(dbg->QueryInterface(IID_PPV_ARGS(&dbg1)))) { dbg1->SetEnableGPUBasedValidation(TRUE); dbg1->Release(); }
            dbg->Release();
            printf("Direct3D 12 debug layer and GPU-based validation are on\n");
        } else {
            fprintf(stderr, "WARNING: the Direct3D 12 debug layer is unavailable (install the Graphics Tools optional feature); continuing without it\n");
        }
        ID3D12DeviceRemovedExtendedDataSettings* dred = nullptr;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dred)))) {
            dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dred->Release();
            printf("Direct3D 12 DRED: auto-breadcrumbs and page faults are on\n");
        }
    }
#endif
    HRESULT hr = CreateDXGIFactory2(factory_flags, IID_PPV_ARGS(&g_factory));
    if (FAILED(hr)) { fprintf(stderr, "ERROR: no Direct3D 12 device: the DXGI factory could not be created (0x%08X)\n", (unsigned)hr); return false; }

    std::vector<IDXGIAdapter1*> adapters;
    for (UINT i = 0; ; i++) {
        IDXGIAdapter1* a = nullptr;
        if (g_factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 d{};
        if (SUCCEEDED(a->GetDesc1(&d)) && (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) { a->Release(); continue; }   // WARP is added once, below, as the last entry
        adapters.push_back(a);
    }
    {   // WARP last, so that every hardware adapter keeps the index DXGI gave it and the software one is one number past
        // the end of them. It is a real device on every Windows and is what the gate's third arm draws on.
        IDXGIAdapter1* warp = nullptr;
        if (SUCCEEDED(g_factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)))) adapters.push_back(warp);
    }
    if (adapters.empty()) { fprintf(stderr, "ERROR: no Direct3D 12 device: DXGI enumerated no adapter at all\n"); return false; }

    std::vector<bool> usable(adapters.size(), false);
    std::vector<std::string> names(adapters.size());
    int first_usable = -1;
    for (size_t i = 0; i < adapters.size(); i++) {
        DXGI_ADAPTER_DESC1 d{};
        const bool got = SUCCEEDED(adapters[i]->GetDesc1(&d));
        names[i] = got ? narrow(d.Description) : std::string("(unnamed adapter)");
        const bool software = got && (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
        usable[i] = SUCCEEDED(D3D12CreateDevice(adapters[i], D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr));
        printf("device %zu: %s (%llu MB, %s)%s\n", i, names[i].c_str(),
               (unsigned long long)(got ? d.DedicatedVideoMemory / (1024 * 1024) : 0), software ? "WARP, software" : "hardware",
               usable[i] ? "" : " - no Direct3D 12 device at feature level 11_0");
        if (usable[i] && first_usable < 0) first_usable = (int)i;
    }
    int chosen = first_usable;
    if (want >= 0) {
        if (want >= (int)adapters.size()) {
            fprintf(stderr, "ERROR: --device %d, and this machine has %zu Direct3D 12 adapter(s)\n", want, adapters.size());
            for (IDXGIAdapter1* a : adapters) a->Release();
            return false;
        }
        if (!usable[(size_t)want]) {
            fprintf(stderr, "ERROR: --device %d (%s) has no Direct3D 12 device at feature level 11_0\n", want, names[(size_t)want].c_str());
            for (IDXGIAdapter1* a : adapters) a->Release();
            return false;
        }
        chosen = want;
    }
    if (chosen < 0) {
        fprintf(stderr, "ERROR: no Direct3D 12 device: none of the %zu adapter(s) offers one at feature level 11_0\n", adapters.size());
#if NNTC_D3D12_LINALG
        // The one way a machine that plainly has a GPU reaches this line. This build exports D3D12SDKVersion, and the
        // runtime then refuses EVERY device - it does not fall back to the system one - unless it finds that version
        // beside the executable. A copy of the .exe on its own is the usual cause, and without this line it reads as a
        // machine with no Direct3D 12 at all.
        fprintf(stderr, "       this build carries the preview redistributable's D3D12SDKVersion export, so the Direct3D 12\n"
                        "       runtime refuses every adapter unless it finds that runtime under D3D12\\ beside this\n"
                        "       executable; check that D3D12\\D3D12Core.dll is there (viewer_d3d12/PHASE_B_NOTES.md section 3)\n");
#endif
        for (IDXGIAdapter1* a : adapters) a->Release();
        return false;
    }
    hr = D3D12CreateDevice(adapters[(size_t)chosen], D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&g_dev));
    for (int i = 0; i < (int)adapters.size(); i++) { if (i == chosen) g_adapter = adapters[(size_t)i]; else adapters[(size_t)i]->Release(); }
    if (FAILED(hr)) { fprintf(stderr, "ERROR: no Direct3D 12 device: D3D12CreateDevice on device %d failed (0x%08X)\n", chosen, (unsigned)hr); return false; }
    printf("drawing on device %d: %s\n", chosen, names[(size_t)chosen].c_str());
    printf("Direct3D 12, feature level 0x%04X\n", (unsigned)D3D_FEATURE_LEVEL_11_0);
#if ENABLE_D3D_DEBUG
    {   // An error from the debug layer stops the run where it happened rather than three frames later. A Debug build of
        // this viewer is expected to be silent under the layer, so a break is the right response to one.
        ID3D12InfoQueue* iq = nullptr;
        if (SUCCEEDED(g_dev->QueryInterface(IID_PPV_ARGS(&iq)))) {
            iq->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
            iq->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
            iq->Release();
        }
    }
#endif
    return true;
}

// ---------------------------------------------------------------------------
// The queue, the command list, the descriptor heaps, the root signature, the eight samplers and the two constant buffers:
// everything that does not depend on the asset or on the frame's size.
// ---------------------------------------------------------------------------
static bool create_common_objects() {
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT; qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL; qd.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE; qd.NodeMask = 0;
    if (FAILED(g_dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&g_queue)))) { fprintf(stderr, "ERROR: the command queue could not be created\n"); return false; }
    for (int i = 0; i < 2; i++)
        if (FAILED(g_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_alloc[i])))) { fprintf(stderr, "ERROR: a command allocator could not be created\n"); return false; }
    if (FAILED(g_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_alloc[0], nullptr, IID_PPV_ARGS(&g_list)))) { fprintf(stderr, "ERROR: the command list could not be created\n"); return false; }
    if (FAILED(g_dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence)))) { fprintf(stderr, "ERROR: the fence could not be created\n"); return false; }
    g_fence_event = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    if (!g_fence_event) { fprintf(stderr, "ERROR: the fence event could not be created\n"); return false; }

    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV; hd.NumDescriptors = 3; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(g_dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_rtv_heap)))) { fprintf(stderr, "ERROR: the RTV heap could not be created\n"); return false; }
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV; hd.NumDescriptors = 1;
    if (FAILED(g_dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_dsv_heap)))) { fprintf(stderr, "ERROR: the DSV heap could not be created\n"); return false; }
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = SRV_HEAP_COUNT; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(g_dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_srv_heap)))) { fprintf(stderr, "ERROR: the SRV heap could not be created\n"); return false; }
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER; hd.NumDescriptors = SAMPLER_HEAP_COUNT;
    if (FAILED(g_dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_samp_heap)))) { fprintf(stderr, "ERROR: the sampler heap could not be created\n"); return false; }
    g_rtv_step = g_dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    g_dsv_step = g_dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    g_srv_step = g_dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    g_samp_step = g_dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

    // Every SRV of the heap is written with a null descriptor first, so that a table bound before an asset is loaded - or
    // one whose third slot has no texture - reads something defined rather than whatever the heap's memory held.
    for (int i = 0; i < SRV_HEAP_COUNT; i++) write_srv(i, nullptr, DXGI_FORMAT_R8G8B8A8_UNORM, 1);
#if NNTC_D3D12_LINALG
    for (int s = 0; s < 3; s++) write_raw_srv(s * SRV_PER_SET + 3, nullptr, 0);   // t3 is a buffer view, not a texture one
#endif

    // The eight sampler states of the Direct3D 11 viewer, each written TWICE so that one table of two descriptors gives
    // s0 and s1 the same state. All CLAMP: the encoder fits the taps clamped at the edges. None of them carries a
    // MipLODBias, because level 1's LOD shift is in the gradients (const0.z) and not in a sampler.
    {
        D3D12_SAMPLER_DESC sd{};
        sd.AddressU = sd.AddressV = sd.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sd.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        sd.MinLOD = 0.0f; sd.MipLODBias = 0.0f;
        for (int i = 0; i < 4; i++) sd.BorderColor[i] = 0.0f;
        static const D3D12_FILTER filters[FILTER_STATES] = { D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT, D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_FILTER_ANISOTROPIC };
        for (int mips = 0; mips < 2; mips++) {   // a second set with MaxLOD = 0: the hardware clamps the LOD to mip 0 (key M), the textures and shader untouched
            sd.MaxLOD = mips ? D3D12_FLOAT32_MAX : 0.0f;
            for (int f = 0; f < FILTER_STATES; f++) {
                sd.Filter = filters[f];
                sd.MaxAnisotropy = f == 3 ? ANISO_MAX : 1;   // anisotropy is several trilinear samples along the footprint's long axis
                const int slot = mips * FILTER_STATES + f;
                g_dev->CreateSampler(&sd, cpu_handle(g_samp_heap, g_samp_step, 2 * slot));
                g_dev->CreateSampler(&sd, cpu_handle(g_samp_heap, g_samp_step, 2 * slot + 1));
            }
        }
    }

    // The root signature: b0 and b1 as root CBVs (the shader's two cbuffers, unchanged, DecCB's float4 W[108] and
    // float4 bias[5] included), one table of three SRVs and one table of two samplers. Version 1.0 through
    // D3D12SerializeRootSignature, so that the 5_0 DXBC binds to it with no RootSignature attribute in the HLSL.
    // The plan's phase B adds one raw SRV at t3 for the fp16 weights buffer and nothing else to this signature.
    {
        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        // Three SRVs, or four in a build with the linear-algebra option: t3 is the weights buffer, which only that
        // build's second pixel shader declares and which the plain DXBC ignores.
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[0].NumDescriptors = (UINT)SRV_PER_SET;
        ranges[0].BaseShaderRegister = 0; ranges[0].RegisterSpace = 0; ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER; ranges[1].NumDescriptors = 2;
        ranges[1].BaseShaderRegister = 0; ranges[1].RegisterSpace = 0; ranges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        D3D12_ROOT_PARAMETER rp[4]{};
        rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; rp[0].Descriptor.ShaderRegister = 0; rp[0].Descriptor.RegisterSpace = 0; rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; rp[1].Descriptor.ShaderRegister = 1; rp[1].Descriptor.RegisterSpace = 0; rp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; rp[2].DescriptorTable.NumDescriptorRanges = 1; rp[2].DescriptorTable.pDescriptorRanges = &ranges[0]; rp[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        rp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; rp[3].DescriptorTable.NumDescriptorRanges = 1; rp[3].DescriptorTable.pDescriptorRanges = &ranges[1]; rp[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC rsd{};
        rsd.NumParameters = 4; rsd.pParameters = rp; rsd.NumStaticSamplers = 0; rsd.pStaticSamplers = nullptr;
        rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ID3DBlob* blob = nullptr, * err = nullptr;
        HRESULT hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err);
        if (FAILED(hr)) {
            fprintf(stderr, "ERROR: the root signature could not be serialised (0x%08X)%s%s\n", (unsigned)hr, err ? ": " : "", err ? (const char*)err->GetBufferPointer() : "");
            safe_release(err); safe_release(blob); return false;
        }
        safe_release(err);
        hr = g_dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&g_rootsig));
        safe_release(blob);
        if (FAILED(hr)) { fprintf(stderr, "ERROR: the root signature could not be created (0x%08X)\n", (unsigned)hr); return false; }
    }

    // The two constant buffers, in an upload heap and mapped once for the program's life. A root CBV wants a 256-byte
    // aligned address, which a committed buffer's start always is, and each one is rounded up to 256 bytes so that the
    // shader can read the whole block it declares.
    const size_t scene_bytes = (sizeof(CBData) + 255) & ~(size_t)255;
    const size_t dec_bytes = (sizeof(DecCB) + 255) & ~(size_t)255;
    g_cb_scene = make_upload_buffer(nullptr, scene_bytes);
    g_cb_dec = make_upload_buffer(nullptr, dec_bytes);
    if (!g_cb_scene || !g_cb_dec) { fprintf(stderr, "ERROR: a constant buffer could not be created\n"); return false; }
    D3D12_RANGE none{ 0, 0 };
    if (FAILED(g_cb_scene->Map(0, &none, (void**)&g_cb_scene_ptr)) || FAILED(g_cb_dec->Map(0, &none, (void**)&g_cb_dec_ptr))) {
        fprintf(stderr, "ERROR: a constant buffer could not be mapped\n"); return false;
    }
    memset(g_cb_scene_ptr, 0, scene_bytes); memset(g_cb_dec_ptr, 0, dec_bytes);
    return true;
}

static bool create_depth(int W, int H) {
    safe_release(g_depth);
    D3D12_RESOURCE_DESC rd = texture_desc(DXGI_FORMAT_D32_FLOAT, W, H, 1);
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE cv{}; cv.Format = DXGI_FORMAT_D32_FLOAT; cv.DepthStencil.Depth = 1.0f; cv.DepthStencil.Stencil = 0;
    const D3D12_HEAP_PROPERTIES def = heap_props(D3D12_HEAP_TYPE_DEFAULT);
    if (FAILED(g_dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_DEPTH_WRITE, &cv, IID_PPV_ARGS(&g_depth)))) {
        fprintf(stderr, "ERROR: the depth buffer could not be created at %dx%d\n", W, H); return false;
    }
    D3D12_DEPTH_STENCIL_VIEW_DESC dv{}; dv.Format = DXGI_FORMAT_D32_FLOAT; dv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D; dv.Flags = D3D12_DSV_FLAG_NONE;
    g_dev->CreateDepthStencilView(g_depth, &dv, g_dsv_heap->GetCPUDescriptorHandleForHeapStart());
    return true;
}

// The frame, recorded into the already-open command list: the Direct3D 11 viewer's own frame with the bindings decided
// first and the descriptor tables in place of its Set*ShaderResources calls.
static void record_scene(D3D12_CPU_DESCRIPTOR_HANDLE rtv, int W, int H) {
    const float clear_color[4] = {0.2f, 0.2f, 0.2f, 1.0f};
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv = g_dsv_heap->GetCPUDescriptorHandleForHeapStart();
    // The frame's bindings are DECIDED FIRST, before the constant buffers are written, because the shader is told in the
    // decoder constant buffer (sel.z) how many textures level 0 is read from. The samplers are picked here too, so keys
    // 4, L and M all take effect on the same frame as the state they change.
    const bool packed = g_bc && g.lat[0].bc_n > 0;
    g.dec.sel[2] = packed ? g.lat[0].bc_n : (g.lat[0].tex_b ? 2 : 0);   // the textures level 0 is read from
    const int slot = (g.mips_on ? FILTER_STATES : 0) + filter_slot();   // key M picks the MaxLOD = 0 set; key X the anisotropic variant
    set_uniforms();

    D3D12_VIEWPORT vp{}; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)W; vp.Height = (float)H; vp.MinDepth = 0.0f; vp.MaxDepth = 1.0f;
    D3D12_RECT scissor{ 0, 0, (LONG)W, (LONG)H };
    g_list->RSSetViewports(1, &vp);
    g_list->RSSetScissorRects(1, &scissor);
    g_list->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
    g_list->ClearRenderTargetView(rtv, clear_color, 0, nullptr);
    g_list->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    ID3D12DescriptorHeap* heaps[2] = { g_srv_heap, g_samp_heap };
    g_list->SetDescriptorHeaps(2, heaps);
    g_list->SetGraphicsRootSignature(g_rootsig);
    g_list->SetGraphicsRootConstantBufferView(0, g_cb_scene->GetGPUVirtualAddress());
    g_list->SetGraphicsRootConstantBufferView(1, g_cb_dec->GetGPUVirtualAddress());
    g_list->SetGraphicsRootDescriptorTable(2, gpu_handle(g_srv_heap, g_srv_step, packed ? SRV_SET_PACKED : SRV_SET_PLAIN));
    g_list->SetGraphicsRootDescriptorTable(3, gpu_handle(g_samp_heap, g_samp_step, 2 * slot));
#if NNTC_D3D12_LINALG
    g_list->SetPipelineState(g_decode == DECODE_LINALG && g.pso_linalg ? g.pso_linalg : g.pso_scene);
#else
    g_list->SetPipelineState(g.pso_scene);
#endif
    g_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    if (g.cube) {
        g_list->IASetVertexBuffers(0, 1, &g_cube_vbv);
        g_list->IASetIndexBuffer(&g_cube_ibv);
        g_list->DrawIndexedInstanced((UINT)g_cube_index_count, 1, 0, 0, 0);
    } else {
        g_list->IASetVertexBuffers(0, 1, &g_quad_vbv);
        g_list->IASetIndexBuffer(&g_quad_ibv);
        g_list->DrawIndexedInstanced(6, 1, 0, 0, 0);
    }
    draw_overlay();
}

static bool flush_uploads() {
    if (!end_commands_and_wait()) { fprintf(stderr, "ERROR: the load-time uploads could not be submitted\n"); return false; }
    for (ID3D12Resource* r : g_upload_keepalive) r->Release();
    g_upload_keepalive.clear();
    return true;
}

// The device, the objects, the shader and the asset, in the order that makes a refusal cost one line: everything that can
// say no is past before a window is shown or a frame is drawn.
static bool init_common(const std::string& json_path, int device_index) {
    if (!create_device(device_index)) return false;
    if (!create_common_objects()) return false;
    // The linear-algebra query, after the command list exists because one of its steps asks that list for a Preview
    // interface. --linalg 1 on a device (or a build) that cannot is a REFUSAL and not a quiet fall back to the plain
    // path: the two paths are compared, and a run that fell back silently would be a measurement of the other one under
    // this one's name. --linalg 0 is always honoured, on every build, because the plain path is the product.
    query_linear_algebra();
    if (g_la_want == 1 && !g_la_available) {
        fprintf(stderr, "ERROR: --linalg 1 was asked for and this device cannot: %s\n", g_la_why_not.c_str());
        return false;
    }
    // The shader travels with the build and is read from beside the executable and from nowhere else (see the note above
    // compile_hlsl_file). The build copies viewer/bin/nntc_view.hlsl there on every build.
    g.shader_path = exe_dir() + "nntc_view.hlsl";
#if NNTC_D3D12_LINALG
    g.linalg_shader_path = exe_dir() + "nntc_view_linalg.hlsl";
#endif
    if (!load_shader(g.shader_path)) return false;
    // load_shader builds the second pipeline, and a Shader Model 6.10 pipeline that did not build turns the query off.
    if (g_la_want == 1 && !g_la_available) {
        fprintf(stderr, "ERROR: --linalg 1 was asked for and this device cannot: %s\n", g_la_why_not.c_str());
        return false;
    }
    if (!load_asset(json_path)) return false;
#if NNTC_D3D12_LINALG
    // The weights buffer and the asset's own range bound, which is the last thing that can refuse the path.
    if (!build_linalg_weights()) return false;
#endif
    if (!flush_uploads()) return false;
    refresh_srv_sets();
    if (!init_overlay()) fprintf(stderr, "WARNING: debug overlay init failed (continuing without it)\n");
    // --tex is given before the asset is read, so the material's texture count is only known now; an out-of-range
    // value would index the shader's output triples past the end.
    if (g.tex_shown < 0 || g.tex_shown >= g.textures_out) { fprintf(stderr, "WARNING: --tex %d is outside 0..%d; showing texture 0\n", g.tex_shown, g.textures_out - 1); g.tex_shown = 0; }
    if (!create_quad((float)g.lat[0].W / (float)g.lat[0].H) || !create_cube()) { fprintf(stderr, "ERROR: the geometry buffers could not be created\n"); return false; }
    // Which path the first frame takes: the linear-algebra one wherever every step passed and --linalg 0 was not given.
    // The reason a run fell back is on the `linear algebra: not available` line above, so this line stays the one word
    // the release gate reads.
    g_decode = (g_la_available && g_la_want != 0) ? DECODE_LINALG : DECODE_PLAIN;
    printf("decode: %s\n", decode_path_name());
    return true;
}

// The frame as a 24-bit bottom-up .bmp, which is what tools/frame_diff.py and the release gate read and what the other two
// viewers write. A shot that was not written is a failed run, with the exit code to say so.
static bool write_bmp(const std::string& path, const uint8_t* rows, size_t row_pitch, int W, int H) {
    const int pitch = (W * 3 + 3) & ~3;
    std::vector<uint8_t> out((size_t)pitch * H, 0);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const uint8_t* sp = rows + (size_t)y * row_pitch + (size_t)x * 4;
        uint8_t* op = &out[(size_t)(H - 1 - y) * pitch + (size_t)x * 3];
        op[0] = sp[2]; op[1] = sp[1]; op[2] = sp[0];
    }
    const uint32_t fsz = 54 + (uint32_t)out.size(); uint8_t hdr[54] = { 'B', 'M' }; uint32_t v;
    v = fsz; memcpy(hdr + 2, &v, 4); v = 54; memcpy(hdr + 10, &v, 4); v = 40; memcpy(hdr + 14, &v, 4); v = (uint32_t)W; memcpy(hdr + 18, &v, 4); v = (uint32_t)H; memcpy(hdr + 22, &v, 4);
    hdr[26] = 1; hdr[28] = 24; v = (uint32_t)out.size(); memcpy(hdr + 34, &v, 4);
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", path.c_str()); return false; }
    const bool ok = fwrite(hdr, 1, 54, f) == 54 && fwrite(out.data(), 1, out.size(), f) == out.size();
    if (fclose(f) != 0 || !ok) { fprintf(stderr, "ERROR: writing %s failed\n", path.c_str()); return false; }
    printf("wrote %s (%dx%d)\n", path.c_str(), W, H);
    return true;
}

// ---------------------------------------------------------------------------
// --shot: ONE frame into an offscreen render target, with no window, no window class and no swap chain at all. The
// Direct3D 11 viewer creates a hidden window because its swap chain needs an HWND; nothing here does, which is what lets
// this --shot run over a remote session and in the release gate without anything appearing on anyone's desktop.
// ---------------------------------------------------------------------------
static int render_shot(const std::string& shot_path, const std::string& json_path, int device_index) {
    if (!init_common(json_path, device_index)) return 1;
    const int W = FRAME_WIDTH, H = FRAME_HEIGHT;
    g_frame_w = W; g_frame_h = H;
    if (!create_depth(W, H)) return 1;

    D3D12_RESOURCE_DESC rd = texture_desc(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 1);
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE cv{}; cv.Format = DXGI_FORMAT_R8G8B8A8_UNORM; cv.Color[0] = 0.2f; cv.Color[1] = 0.2f; cv.Color[2] = 0.2f; cv.Color[3] = 1.0f;
    const D3D12_HEAP_PROPERTIES def = heap_props(D3D12_HEAP_TYPE_DEFAULT);
    if (FAILED(g_dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_RENDER_TARGET, &cv, IID_PPV_ARGS(&g_shot_target)))) {
        fprintf(stderr, "ERROR: --shot could not create the %dx%d render target\n", W, H); return 1;
    }
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = cpu_handle(g_rtv_heap, g_rtv_step, 2);
    D3D12_RENDER_TARGET_VIEW_DESC rv{}; rv.Format = DXGI_FORMAT_R8G8B8A8_UNORM; rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    g_dev->CreateRenderTargetView(g_shot_target, &rv, rtv);

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{}; UINT rows = 0; UINT64 row_bytes = 0, total = 0;
    g_dev->GetCopyableFootprints(&rd, 0, 1, 0, &fp, &rows, &row_bytes, &total);
    const D3D12_HEAP_PROPERTIES rb = heap_props(D3D12_HEAP_TYPE_READBACK);
    const D3D12_RESOURCE_DESC rbd = buffer_desc(total);
    ID3D12Resource* readback = nullptr;
    if (FAILED(g_dev->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &rbd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)))) {
        fprintf(stderr, "ERROR: --shot could not create the readback buffer\n"); return 1;
    }
    if (!begin_commands()) { fprintf(stderr, "ERROR: --shot could not record its frame\n"); readback->Release(); return 1; }
    upload_overlay_if_dirty();
    record_scene(rtv, W, H);
    barrier(g_shot_target, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    {
        D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource = readback; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = fp;
        D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource = g_shot_target; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; src.SubresourceIndex = 0;
        g_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    if (!end_commands_and_wait()) { fprintf(stderr, "ERROR: --shot could not submit its frame\n"); readback->Release(); return 1; }

    uint8_t* mapped = nullptr;
    D3D12_RANGE whole{ 0, (SIZE_T)total };
    if (FAILED(readback->Map(0, &whole, (void**)&mapped))) { fprintf(stderr, "ERROR: --shot could not map the readback buffer\n"); readback->Release(); return 1; }
    const bool ok = write_bmp(shot_path, mapped + fp.Offset, (size_t)fp.Footprint.RowPitch, W, H);
    D3D12_RANGE none{ 0, 0 };
    readback->Unmap(0, &none);
    readback->Release();
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// --bench: N headless frames, timed with two timestamp queries around the scene draw, and nothing written.
//
// INFORMATIONAL ONLY, and said so in the usage as well. This viewer is a correctness sample: what the two decode paths
// have to agree on is the PICTURE, and --bench exists so that a number can be quoted beside a run rather than estimated.
// The mean and the minimum both, because the minimum is the least disturbed frame and the mean is what a reader would
// otherwise compute wrongly from one number; neither is asserted anywhere.
// ---------------------------------------------------------------------------
static int render_bench(int frames, const std::string& json_path, int device_index) {
    if (!init_common(json_path, device_index)) return 1;
    const int W = FRAME_WIDTH, H = FRAME_HEIGHT;
    g_frame_w = W; g_frame_h = H;
    if (!create_depth(W, H)) return 1;

    UINT64 frequency = 0;
    if (FAILED(g_queue->GetTimestampFrequency(&frequency)) || !frequency) {
        fprintf(stderr, "ERROR: --bench: this queue reports no timestamp frequency, so a frame here cannot be timed\n");
        return 1;
    }
    D3D12_RESOURCE_DESC rd = texture_desc(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 1);
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE cv{}; cv.Format = DXGI_FORMAT_R8G8B8A8_UNORM; cv.Color[0] = 0.2f; cv.Color[1] = 0.2f; cv.Color[2] = 0.2f; cv.Color[3] = 1.0f;
    const D3D12_HEAP_PROPERTIES def = heap_props(D3D12_HEAP_TYPE_DEFAULT);
    if (FAILED(g_dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_RENDER_TARGET, &cv, IID_PPV_ARGS(&g_shot_target)))) {
        fprintf(stderr, "ERROR: --bench could not create the %dx%d render target\n", W, H); return 1;
    }
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = cpu_handle(g_rtv_heap, g_rtv_step, 2);
    D3D12_RENDER_TARGET_VIEW_DESC rv{}; rv.Format = DXGI_FORMAT_R8G8B8A8_UNORM; rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    g_dev->CreateRenderTargetView(g_shot_target, &rv, rtv);

    D3D12_QUERY_HEAP_DESC qh{}; qh.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; qh.Count = 2; qh.NodeMask = 0;
    ID3D12QueryHeap* queries = nullptr;
    if (FAILED(g_dev->CreateQueryHeap(&qh, IID_PPV_ARGS(&queries)))) { fprintf(stderr, "ERROR: --bench could not create the timestamp query heap\n"); return 1; }
    const D3D12_HEAP_PROPERTIES rb = heap_props(D3D12_HEAP_TYPE_READBACK);
    const D3D12_RESOURCE_DESC rbd = buffer_desc(2 * sizeof(UINT64));
    ID3D12Resource* readback = nullptr;
    if (FAILED(g_dev->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &rbd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)))) {
        fprintf(stderr, "ERROR: --bench could not create the readback buffer\n"); queries->Release(); return 1;
    }
    double total = 0.0, best = 0.0;
    for (int i = 0; i < frames; i++) {
        if (!begin_commands()) { fprintf(stderr, "ERROR: --bench could not record a frame\n"); readback->Release(); queries->Release(); return 1; }
        upload_overlay_if_dirty();
        g_list->EndQuery(queries, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        record_scene(rtv, W, H);
        g_list->EndQuery(queries, D3D12_QUERY_TYPE_TIMESTAMP, 1);
        g_list->ResolveQueryData(queries, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, readback, 0);
        if (!end_commands_and_wait()) { fprintf(stderr, "ERROR: --bench could not submit a frame\n"); readback->Release(); queries->Release(); return 1; }
        UINT64* t = nullptr;
        D3D12_RANGE whole{ 0, 2 * sizeof(UINT64) };
        if (FAILED(readback->Map(0, &whole, (void**)&t))) { fprintf(stderr, "ERROR: --bench could not map the readback buffer\n"); readback->Release(); queries->Release(); return 1; }
        const double us = (double)(t[1] - t[0]) * 1000000.0 / (double)frequency;
        D3D12_RANGE none{ 0, 0 };
        readback->Unmap(0, &none);
        total += us;
        if (i == 0 || us < best) best = us;
    }
    printf("bench: %d frames at %dx%d, decode %s: mean %.1f us, min %.1f us per scene draw (informational)\n",
           frames, W, H, decode_path_name(), total / (double)frames, best);
    readback->Release(); queries->Release();
    return 0;
}

// ---------------------------------------------------------------------------
// The window path: a flip-model swap chain of two buffers on a Win32 window. The window is created HIDDEN and shown only
// once the device, the asset and the shader are all past refusing, so a run that exits 1 never flashes a window up.
// ---------------------------------------------------------------------------
static bool create_backbuffer_views() {
    for (int i = 0; i < 2; i++) {
        safe_release(g_backbuffer[i]);
        if (FAILED(g_swapchain->GetBuffer((UINT)i, IID_PPV_ARGS(&g_backbuffer[i])))) return false;
        D3D12_RENDER_TARGET_VIEW_DESC rv{}; rv.Format = DXGI_FORMAT_R8G8B8A8_UNORM; rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        g_dev->CreateRenderTargetView(g_backbuffer[i], &rv, cpu_handle(g_rtv_heap, g_rtv_step, i));
    }
    return true;
}
static void resize_swapchain(int w, int h) {
    if (!g_swapchain || w <= 0 || h <= 0) return;
    if (w == WINDOW_WIDTH && h == WINDOW_HEIGHT && g_backbuffer[0]) return;   // the size it already has: nothing to do, and no frame to lose
    WINDOW_WIDTH = w; WINDOW_HEIGHT = h;
    wait_for_gpu();   // the back buffers cannot be released while a frame is still reading them
    for (int i = 0; i < 2; i++) safe_release(g_backbuffer[i]);
    if (FAILED(g_swapchain->ResizeBuffers(2, (UINT)w, (UINT)h, DXGI_FORMAT_R8G8B8A8_UNORM, 0))) {
        fprintf(stderr, "ERROR: the swap chain could not be resized to %dx%d\n", w, h); return;
    }
    if (!create_backbuffer_views()) fprintf(stderr, "ERROR: backbuffer view recreation failed after resize\n");
    if (!create_depth(w, h)) fprintf(stderr, "ERROR: the depth buffer could not be recreated after resize\n");
    g.debug_dirty = true;
}
static int render_window(const std::string& json_path, int device_index) {
    WNDCLASSEXA wc{}; wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW; wc.lpfnWndProc = wnd_proc; wc.hInstance = GetModuleHandleA(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW); wc.lpszClassName = "nntc_view_d3d12_wc";
    RegisterClassExA(&wc);
    RECT r{0,0,WINDOW_WIDTH,WINDOW_HEIGHT};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowA(wc.lpszClassName, "NNTC latent viewer (D3D12)", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                              r.right - r.left, r.bottom - r.top, nullptr, nullptr, wc.hInstance, nullptr);
    if (!hwnd) { fprintf(stderr, "ERROR: window creation failed\n"); return 1; }
    // The size the window ACTUALLY got: Windows clamps a window to the work area, so on a smaller desktop the client rect
    // is not what was asked for, and the swap chain is created at the client size so that nothing is stretched on Present.
    { RECT cr{}; if (GetClientRect(hwnd, &cr) && cr.right > cr.left && cr.bottom > cr.top) { WINDOW_WIDTH = (int)(cr.right - cr.left); WINDOW_HEIGHT = (int)(cr.bottom - cr.top); } }

    if (!init_common(json_path, device_index)) { DestroyWindow(hwnd); return 1; }
    printf("window: %dx%d client area\n", WINDOW_WIDTH, WINDOW_HEIGHT);

    // Non-sRGB back buffer and UNORM textures throughout: the latents are not colours, and the decoder's output is
    // written as it is.
    DXGI_SWAP_CHAIN_DESC1 scd{};
    scd.Width = (UINT)WINDOW_WIDTH; scd.Height = (UINT)WINDOW_HEIGHT; scd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.Stereo = FALSE; scd.SampleDesc.Count = 1; scd.SampleDesc.Quality = 0;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; scd.BufferCount = 2; scd.Scaling = DXGI_SCALING_STRETCH;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD; scd.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED; scd.Flags = 0;
    IDXGISwapChain1* sc1 = nullptr;
    HRESULT hr = g_factory->CreateSwapChainForHwnd(g_queue, hwnd, &scd, nullptr, nullptr, &sc1);
    if (FAILED(hr)) { fprintf(stderr, "ERROR: the swap chain could not be created (0x%08X)\n", (unsigned)hr); DestroyWindow(hwnd); return 1; }
    hr = sc1->QueryInterface(IID_PPV_ARGS(&g_swapchain));
    sc1->Release();
    if (FAILED(hr)) { fprintf(stderr, "ERROR: IDXGISwapChain3 is not available (0x%08X)\n", (unsigned)hr); DestroyWindow(hwnd); return 1; }
    g_factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
    if (!create_backbuffer_views() || !create_depth(WINDOW_WIDTH, WINDOW_HEIGHT)) { fprintf(stderr, "ERROR: the back buffer views could not be created\n"); DestroyWindow(hwnd); return 1; }

    ShowWindow(hwnd, SW_SHOW);
    LARGE_INTEGER qpf, last, now; QueryPerformanceFrequency(&qpf); QueryPerformanceCounter(&last);
    while (!g_quit) {
        MSG msg;
        while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageA(&msg); }
        if (g_quit) break;
        QueryPerformanceCounter(&now);
        float dt = (float)((double)(now.QuadPart - last.QuadPart) / (double)qpf.QuadPart);
        last = now;
        if (dt > 0.1f) dt = 0.1f;
        process_held_keys(hwnd, dt);
        if (!g_backbuffer[0] || !g_depth) continue;
        g_frame_w = WINDOW_WIDTH; g_frame_h = WINDOW_HEIGHT;
        const int bb = (int)g_swapchain->GetCurrentBackBufferIndex();
        if (!begin_commands()) { fprintf(stderr, "ERROR: the frame's commands could not be recorded\n"); break; }
        upload_overlay_if_dirty();
        barrier(g_backbuffer[bb], D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
        record_scene(cpu_handle(g_rtv_heap, g_rtv_step, bb), WINDOW_WIDTH, WINDOW_HEIGHT);
        barrier(g_backbuffer[bb], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
        if (!end_commands_and_wait()) { fprintf(stderr, "ERROR: the frame could not be submitted\n"); break; }
        g_swapchain->Present(1, 0);
    }
    wait_for_gpu();
    DestroyWindow(hwnd);
    return 0;
}

static void destroy_everything() {
    if (g_queue && g_fence) wait_for_gpu();
    for (ID3D12Resource* r : g_upload_keepalive) r->Release();
    g_upload_keepalive.clear();
    for (int l = 0; l < 2; l++) {
        safe_release(g.lat[l].tex); safe_release(g.lat[l].tex_b);
        for (int p = 0; p < 2; p++) safe_release(g.lat[l].bc_tex[p]);
    }
    safe_release(g.debug_tex); safe_release(g.debug_upload); safe_release(g.debug_vb); safe_release(g.debug_ib);
    safe_release(g_quad_vb); safe_release(g_quad_ib); safe_release(g_cube_vb); safe_release(g_cube_ib);
    safe_release(g.pso_scene); safe_release(g.pso_overlay);
#if NNTC_D3D12_LINALG
    safe_release(g.pso_linalg); safe_release(g_la_buffer);
#endif
    safe_release(g_cb_scene); safe_release(g_cb_dec);
    safe_release(g_shot_target); safe_release(g_depth);
    for (int i = 0; i < 2; i++) safe_release(g_backbuffer[i]);
    safe_release(g_swapchain);
    safe_release(g_rootsig);
    safe_release(g_srv_heap); safe_release(g_samp_heap); safe_release(g_rtv_heap); safe_release(g_dsv_heap);
    safe_release(g_list);
    for (int i = 0; i < 2; i++) safe_release(g_alloc[i]);
    safe_release(g_fence);
    if (g_fence_event) { CloseHandle(g_fence_event); g_fence_event = nullptr; }
    safe_release(g_queue);
    safe_release(g_dev);
    safe_release(g_adapter);
    safe_release(g_factory);
}

int main(int argc, char** argv) {
#ifndef NDEBUG
    printf("DEBUG build\n");   // the first line out, before any argument is read, so the build type is never in doubt
#endif
    std::string json_path, shot_path;
    int device_index = -1;   // -1 = the first adapter with a Direct3D 12 device; --device N overrides it
    for (int i = 1; i < argc; i++) { std::string a = argv[i];
        if (a == "--help" || a == "-h") { viewer_usage(argv[0]); return 0; }   // asking for the usage is not a refusal
        else if (a == "--shot") { if (i + 1 >= argc) { fprintf(stderr, "ERROR: --shot needs a file name\n"); return 1; } shot_path = argv[++i]; g_shot = true; }
        else if (a == "--cube") g.cube = true;
        else if (a == "--nomips") g.mips_on = false;   // the key M state for --shot
        else if (a == "--nobias") g.lod_bias = false;   // the key L state for --shot
        else if (a == "--noaniso") g.aniso = false;   // the key X state for --shot
        else if (a == "--renorm") g.const0[3] = 1.0f;   // the key V state for --shot
        else if (a == "--raw0") g.const0[0] = 1.0f;
        else if (a == "--raw1") g.const0[1] = 1.0f;
        else if (a == "--bc") g_bc = true;
        else if (a == "--nooverlay") g_overlay = false;
        else if (a == "--z" || a == "--yaw" || a == "--pitch") {
            double v; if (i + 1 >= argc || !parse_number(argv[i + 1], v)) { fprintf(stderr, "ERROR: %s needs a number, not '%s'\n", a.c_str(), i + 1 < argc ? argv[i + 1] : ""); return 1; }
            i++; if (a == "--z") g.z = (float)v; else if (a == "--yaw") g.yaw = (float)v; else g.pitch = (float)v; }
        else if (a == "--tex") {
            int v; if (i + 1 >= argc || !parse_integer(argv[i + 1], v)) { fprintf(stderr, "ERROR: --tex needs a texture index, not '%s'\n", i + 1 < argc ? argv[i + 1] : ""); return 1; }
            i++; g.tex_shown = v; }
        else if (a == "--size") {
            int w = 0, h = 0;
            if (i + 2 >= argc || !parse_integer(argv[i + 1], w) || !parse_integer(argv[i + 2], h) || w < 16 || h < 16 || w > 16384 || h > 16384) {
                fprintf(stderr, "ERROR: --size needs a width and a height in 16..16384, not '%s %s'\n",
                        i + 1 < argc ? argv[i + 1] : "", i + 2 < argc ? argv[i + 2] : "");
                return 1;
            }
            i += 2; FRAME_WIDTH = w; FRAME_HEIGHT = h; WINDOW_WIDTH = w; WINDOW_HEIGHT = h; }
        else if (a == "--linalg") {
            int v; if (i + 1 >= argc || !parse_integer(argv[i + 1], v) || (v != 0 && v != 1)) {
                fprintf(stderr, "ERROR: --linalg needs 0 or 1, not '%s'\n", i + 1 < argc ? argv[i + 1] : "");
                return 1;
            }
            i++; g_la_want = v; }
        else if (a == "--linalg-layout") {
            const std::string v = i + 1 < argc ? argv[i + 1] : "";
            if (v != "row" && v != "optimal") {
                fprintf(stderr, "ERROR: --linalg-layout needs row or optimal, not '%s'\n", v.c_str());
                return 1;
            }
            i++; g_la_layout_want = v == "row" ? 0 : 1; }
        else if (a == "--bench") {
            int v; if (i + 1 >= argc || !parse_integer(argv[i + 1], v) || v < 1 || v > 100000) {
                fprintf(stderr, "ERROR: --bench needs a frame count in 1..100000, not '%s'\n", i + 1 < argc ? argv[i + 1] : "");
                return 1;
            }
            i++; g_bench_frames = v; }
        else if (a == "--device") {
            if (i + 1 >= argc || !parse_integer(argv[i + 1], device_index) || device_index < 0) {
                fprintf(stderr, "ERROR: --device needs a device index, not '%s'\n", i + 1 < argc ? argv[i + 1] : "");
                return 1;
            }
            i++; }
        else if (!a.empty() && a[0] == '-') { fprintf(stderr, "ERROR: unknown option '%s'\n", a.c_str()); viewer_usage(argv[0]); return 1; }
        else if (json_path.empty()) json_path = a;
        else { fprintf(stderr, "ERROR: one descriptor only; '%s' is a second\n", a.c_str()); return 1; } }
    // --bench writes no file and --shot writes exactly one, so asking for both is asking for two different runs: the
    // timing loop would be measuring frames nobody keeps and the shot would be one of them. The Vulkan viewer refuses
    // the same pair for the same reason.
    if (g_bench_frames && g_shot) {
        fprintf(stderr, "ERROR: --bench and --shot are two different runs: one times frames and writes nothing, the "
                        "other writes one frame and times nothing\n");
        return 1;
    }
    // Nothing is printed about the run's state before it is known that there IS a run: named no asset, the program owes
    // the caller the usage and nothing else.
    if (json_path.empty()) {
        viewer_usage(argv[0]);
        return 1;   // no asset was named, which IS a refusal, unlike --help above
    }
    // The descriptor is opened here, before a device or a shader exists, so that a path this program cannot read costs one
    // line and an exit. "cannot read" would cover three different mistakes, because an empty read is all three, so the
    // caller is told which: a path that is not there, a directory named where a descriptor was meant, and a file that
    // exists but holds nothing.
    {
        std::error_code ec;
        const std::filesystem::path p(json_path);
        if (!std::filesystem::exists(p, ec)) { fprintf(stderr, "ERROR: cannot read '%s'\n", json_path.c_str()); return 1; }
        if (!std::filesystem::is_regular_file(p, ec)) { fprintf(stderr, "ERROR: '%s' is not a file\n", json_path.c_str()); return 1; }
        if (read_file(json_path).empty()) { fprintf(stderr, "ERROR: '%s' is empty\n", json_path.c_str()); return 1; }
    }
    // --shot's destination, checked before a device exists. A directory that is not there is a typo, and a typo is worth
    // one line now rather than after the whole asset has been read and drawn; the write itself keeps its own checks.
    if (g_shot) {
        const std::filesystem::path out(shot_path);
        const std::filesystem::path dir = out.has_parent_path() ? out.parent_path() : std::filesystem::path(".");
        std::error_code ec;
        if (!std::filesystem::is_directory(dir, ec)) {
            fprintf(stderr, "ERROR: cannot write '%s': '%s' is not a directory this program can see\n", shot_path.c_str(), dir.string().c_str());
            return 1;
        }
    }
    // `level 0` here is what the VIEWER will bind, which for a file that is already block-compressed is the file's own
    // blocks and not a pack of this program's making; load_asset says which, once it has read the file.
    printf("start: mips %s, level 1 LOD shift %s, anisotropy %s, level 0 %s\n", g.mips_on ? "on" : "off", g.lod_bias ? "on" : "off",
           g.aniso ? "on (trilinear only)" : "off", g_bc ? "the BC pack, if the file is uncompressed" : "as the file holds it");
    const int status = g_bench_frames ? render_bench(g_bench_frames, json_path, device_index)
                     : g_shot          ? render_shot(shot_path, json_path, device_index)
                                       : render_window(json_path, device_index);
    destroy_everything();
    if (!g_shot && !g_bench_frames && status == 0) printf("Done.\n");
    return status;
}
