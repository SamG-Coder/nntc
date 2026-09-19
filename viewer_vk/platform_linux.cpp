// nntc_view_vk: the Linux window, through GLFW 3 - the small C library every Vulkan sample uses for its window,
// which speaks Wayland and X11 natively and chooses at run time - and, when the build found no GLFW, the headless
// block under which everything but --shot refuses by name.
#ifndef _WIN32

#include "platform.h"
#include "vk_check.h"
#include <cstdio>
#include <cstdint>

#ifdef NNTC_GLFW

// The GLFW window. GLFW is initialised here and not in main because the platform layer owns the window system,
// and glfw3.h is included after vulkan.h so that it declares glfwCreateWindowSurface and the extension query.
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

static GLFWwindow* g_window = nullptr;

// GLFW's own diagnostics, printed as they arrive: a GLFW call that fails also returns its failure, which is what
// the code below acts on, so this only adds the reason to the ERROR line that follows it.
static void glfw_error(int code, const char* text) { fprintf(stderr, "GLFW error %d: %s\n", code, text); }

// The translation from GLFW's key codes to the viewer's own. A letter or a digit needs none - GLFW names those by
// their ASCII code, as Win32 does - and both shifts are the viewer's one Shift.
static const struct { int app, glfw; } GLFW_KEY_MAP[] = {
    { KEY_ESCAPE, GLFW_KEY_ESCAPE }, { KEY_SPACE, GLFW_KEY_SPACE }, { KEY_LEFT, GLFW_KEY_LEFT },
    { KEY_RIGHT, GLFW_KEY_RIGHT }, { KEY_UP, GLFW_KEY_UP }, { KEY_DOWN, GLFW_KEY_DOWN },
    { KEY_SHIFT, GLFW_KEY_LEFT_SHIFT }, { KEY_SHIFT, GLFW_KEY_RIGHT_SHIFT }, { KEY_F1, GLFW_KEY_F1 },
};

static int glfw_to_app_key(int key) {
    for (const auto& e : GLFW_KEY_MAP) if (e.glfw == key) return e.app;
    return (key >= '0' && key <= '9') || (key >= 'A' && key <= 'Z') ? key : KEY_OTHER;
}

static void glfw_key(GLFWwindow*, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;   // GLFW_REPEAT is a key already down, and a release is not a strike
    if (g_shot) return;                 // the shot frame is a function of the command line alone
    app_key_struck(glfw_to_app_key(key));   // every key, the unused ones as KEY_OTHER: any key closes the help page
}

// The size is not acted on here, for the Win32 block's reason: the callback only raises the flag the frame loop
// reads, and the loop recreates the swapchain between frames and not in the middle of one.
static void glfw_framebuffer_size(GLFWwindow*, int, int) { g_resized = true; }

// The live state of one key, from GLFW's own per-window table, and only while this window has the focus: GLFW
// synthesises the releases when the focus leaves, so a key that went up in another window is never held here.
bool win_key_held(int key) {
    if (!g_window || !glfwGetWindowAttrib(g_window, GLFW_FOCUSED)) return false;
    if (key == KEY_SHIFT)
        return glfwGetKey(g_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
               glfwGetKey(g_window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    for (const auto& e : GLFW_KEY_MAP) if (e.app == key) return glfwGetKey(g_window, e.glfw) == GLFW_PRESS;
    return glfwGetKey(g_window, key) == GLFW_PRESS;
}

bool win_open(int w, int h, const char* title) {
    glfwSetErrorCallback(glfw_error);
    if (!glfwInit()) {
        fprintf(stderr, "ERROR: GLFW could not initialise; is there a display (DISPLAY or WAYLAND_DISPLAY)? "
                        "--shot needs none\n");
        return false;
    }
    if (!glfwVulkanSupported()) {
        fprintf(stderr, "ERROR: GLFW found no Vulkan loader; install the Vulkan loader (libvulkan1 or equivalent)\n");
        return false;
    }
    // Windows clamps a window to the WORK AREA - the screen less its panels - and the swapchain is made from what it
    // actually got; the same rule is applied here before the window exists, so the default 2560x1440 on a 1080p
    // laptop becomes a window that fits above the panel rather than one whose bottom sits under it. The work area
    // is in screen units, as the size given to glfwCreateWindow is.
    if (GLFWmonitor* monitor = glfwGetPrimaryMonitor()) {
        int ax = 0, ay = 0, aw = 0, ah = 0;
        glfwGetMonitorWorkarea(monitor, &ax, &ay, &aw, &ah);
        if (aw > 0 && w > aw) w = aw;
        if (ah > 0 && h > ah) h = ah;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);   // Vulkan draws into it; GLFW makes no GL context
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);       // win_show puts it on the screen, once nothing can refuse
    g_window = glfwCreateWindow(w, h, title, nullptr, nullptr);
    if (!g_window) { fprintf(stderr, "ERROR: window creation failed\n"); return false; }
    glfwSetKeyCallback(g_window, glfw_key);
    glfwSetFramebufferSizeCallback(g_window, glfw_framebuffer_size);
    return true;
}

void win_show(void) {
    // No glfwFocusWindow: GLFW_FOCUS_ON_SHOW is on by default, so showing the window focuses it wherever focus can
    // be given, and Wayland - which lets no client take the focus - would print a GLFW error for the request on
    // every start-up, a line that reads as a failure and is not one.
    if (g_window) glfwShowWindow(g_window);
}

void win_pump(void) { glfwPollEvents(); }

bool win_should_quit(void) { return (g_window && glfwWindowShouldClose(g_window)) || g_quit_requested; }

// The framebuffer size in PIXELS, which on a scaled desktop is not the window size in screen units; the swapchain
// is created from this, as on Windows it is created from the client rect.
void win_client_size(int* w, int* h) {
    if (g_window) glfwGetFramebufferSize(g_window, w, h); else { *w = 0; *h = 0; }
}

VkSurfaceKHR win_create_surface(VkInstance instance) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VK_CHECK(glfwCreateWindowSurface(instance, g_window, nullptr, &surface));
    return surface;
}

// GLFW names the extensions for the backend it chose - VK_KHR_wayland_surface, VK_KHR_xcb_surface or another - with
// VK_KHR_surface among them, which is why the list is asked for rather than written down.
bool win_instance_extensions(std::vector<const char*>& out) {
    uint32_t n = 0;
    const char** names = glfwGetRequiredInstanceExtensions(&n);
    if (!names || n == 0) return false;
    for (uint32_t i = 0; i < n; i++) out.push_back(names[i]);
    return true;
}

double win_seconds(void) { return glfwGetTime(); }

// glfwWaitEvents returns on the next event of any kind - a restore, a resize, a close - which is WaitMessage's
// behaviour: the loop wakes on exactly the events that could give it something to draw into.
void win_idle(void) { glfwWaitEvents(); }

#else

// A build with no window system at all - a Linux configure that found Vulkan and no GLFW. Everything except --shot
// refuses, and --shot is exactly the mode that needs nothing from this block.
bool win_instance_extensions(std::vector<const char*>&) { return false; }

bool win_open(int, int, const char*) {
    fprintf(stderr, "ERROR: this build has no window system, so only --shot runs here: CMake found no GLFW when it "
                    "configured. Install libglfw3-dev (glfw-devel, libglfw-devel or glfw elsewhere), run cmake "
                    "again and rebuild\n");
    return false;
}
void win_show(void) {}   // there is no window to show, and win_open refused before this could be called
// No window means no key to deliver, and naming the handler is what keeps it from being a static function nothing
// in this build references, which clang reports.
void win_pump(void) { (void)app_key_struck; }
bool win_should_quit(void) { return true; }
void win_client_size(int* w, int* h) { *w = 0; *h = 0; }
VkSurfaceKHR win_create_surface(VkInstance) {
    fprintf(stderr, "ERROR: this build has no window system, so only --shot runs here: CMake found no GLFW when it "
                    "configured. Install libglfw3-dev (glfw-devel, libglfw-devel or glfw elsewhere), run cmake "
                    "again and rebuild\n");
    return VK_NULL_HANDLE;
}
double win_seconds(void) { return 0.0; }
bool win_key_held(int) { return false; }   // no window, so no key is down in one
void win_idle(void) {}                     // and no window to wait on: win_open refuses before the loop starts


#endif   // NNTC_GLFW, else headless

#endif   // not _WIN32
