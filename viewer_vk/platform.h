// nntc_view_vk: the platform layer's interface. The window, its events, its size, its surface, the clock and the
// live state of a key, as ten functions main.cpp calls and one enum both sides speak; the bodies are in
// platform_win32.cpp and platform_linux.cpp, one of which the build compiles.
#pragma once
#include <vulkan/vulkan.h>
#include <vector>

// The three flags shared across the boundary, defined in main.cpp. The window raises the first (a resize the frame
// loop turns into one swapchain recreate) and reads the other two: --shot, under which a keystroke changes nothing,
// and Esc, which the key handling raises because which key quits is the viewer's business and not the platform's.
extern bool g_resized;
extern bool g_shot;
extern bool g_quit_requested;

// ---------------------------------------------------------------------------
// The platform layer: eight functions, and one constant naming the instance extension the surface needs. One of them
// is stage 3's: a key held down is a state to be scanned once a frame and not a message, which is how both viewers
// move the camera smoothly.
//
// The Win32 bodies are the Direct3D viewer's, lifted so that the two viewers' window behaviour is identical by
// construction rather than by intention - the client rect read back after CreateWindow, the message pump, and the rule
// that --shot touches no input. The GLFW block beside them does the same things for Linux through GLFW 3, the
// small C library every Vulkan sample uses for its window: it speaks Wayland and X11 natively and chooses at run
// time, so a user of either build finds the same window, the same keys and the same behaviour on a resize, a
// minimise and a close. The Windows block stays native on purpose: it is the Direct3D viewer's code, the two
// viewers behave identically there by construction, and the Windows build needs nothing fetched to configure.
// ---------------------------------------------------------------------------
bool win_open(int w, int h, const char* title);
// The window is CREATED by win_open and SHOWN by this, separately and on purpose: every refusal between the two - a
// device that cannot do what a flag asked for, an asset that will not load, a shader that will not compile - then
// prints its ERROR and exits without a window ever having appeared. A window that flashes up and vanishes reads as a
// crash; nothing was drawn into it in any case, because it is shown only once the first pipeline exists.
void win_show(void);
void win_pump(void);
bool win_should_quit(void);
void win_client_size(int* w, int* h);
VkSurfaceKHR win_create_surface(VkInstance instance);
// The instance extensions the window's surface needs, VK_KHR_surface included. They are the window system's to
// name rather than a constant of the build, because GLFW's depend on which backend it chose at run time; false
// means this build has no window system at all.
bool win_instance_extensions(std::vector<const char*>& out);
double win_seconds(void);
bool win_key_held(int key);   // one key's live state, and false whenever the window is not the foreground one
// Block until the window system has something to say. It is called only while there is nothing to draw into - a
// minimised window, or one dragged down to no client area at all - so that the frame loop waits for the restore
// instead of spinning a core on a swapchain it cannot create.
void win_idle(void);

// The keys the viewer knows, as the program's own codes rather than the platform's, so that the key handling is
// written once and each platform block translates into it. A letter or a digit IS its ASCII character - which is also
// its Win32 virtual-key code, so that half of the translation is the identity - and the named keys take values below
// 32, which no character uses.
enum AppKey { KEY_ESCAPE = 1, KEY_SPACE, KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, KEY_SHIFT };

// The two functions the platform blocks call into: one struck key, and one frame's held-key scan. They are defined far
// below, beside the state they change, because what a key means is not a windowing concern.
void app_key_struck(int key);

