// nntc_view_vk: the Win32 window. These are the Direct3D viewer's bodies, lifted so that the two viewers' window
// behaviour is identical by construction rather than by intention - the client rect read back after CreateWindow,
// the message pump, and the rule that --shot touches no input. Native on purpose: the Windows build needs nothing
// fetched to configure, and this file is what keeps it that way.
#ifdef _WIN32

// vulkan.h pulls windows.h in for the Win32 surface, so the two macros that keep that header small and keep it from
// defining min and max as macros have to be set before it.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "platform.h"
#include "vk_check.h"
#include <cstdio>


#include <windows.h>

bool win_instance_extensions(std::vector<const char*>& out) {
    out.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    out.push_back("VK_KHR_win32_surface");
    return true;
}

static bool g_quit = false;
static HWND g_hwnd = nullptr;
static HINSTANCE g_hinstance = nullptr;

// The translation both ways between Win32's virtual-key codes and the viewer's own. A letter or a digit needs none:
// Win32 gives those the ASCII code already, which is why the table below names only the eight that differ.
static const struct { int app, vk; } WIN32_KEY_MAP[] = {
    { KEY_ESCAPE, VK_ESCAPE }, { KEY_SPACE, VK_SPACE }, { KEY_LEFT, VK_LEFT }, { KEY_RIGHT, VK_RIGHT },
    { KEY_UP, VK_UP }, { KEY_DOWN, VK_DOWN }, { KEY_SHIFT, VK_SHIFT }, { KEY_F1, VK_F1 },
};

static int win32_to_app_key(int vk) {
    for (const auto& e : WIN32_KEY_MAP) if (e.vk == vk) return e.app;
    return (vk >= '0' && vk <= '9') || (vk >= 'A' && vk <= 'Z') ? vk : KEY_OTHER;
}

static int app_key_to_win32(int key) {
    for (const auto& e : WIN32_KEY_MAP) if (e.app == key) return e.vk;
    return key;
}

static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_CLOSE: case WM_DESTROY: g_quit = true; PostQuitMessage(0); return 0;
        // The size is not acted on here: a swapchain cannot be recreated from inside the window procedure while the
        // frame loop may be halfway through a frame, so the message only raises a flag the loop reads.
        case WM_SIZE: if (wp != SIZE_MINIMIZED) g_resized = true; return 0;
        case WM_SYSKEYDOWN:   // Alt and F10 arrive here: they close the help page too, and otherwise go to DefWindowProc as before
            if (!g_shot && app_help_open() && !(lp & (1 << 30))) { app_key_struck(win32_to_app_key((int)wp)); return 0; }
            break;
        case WM_KEYDOWN:
            if (g_shot) return 0;   // the shot frame is a function of the command line, so a keystroke changes nothing
            if (lp & (1 << 30)) return 0;   // a repeat of a key already down is not a fresh press
            app_key_struck(win32_to_app_key((int)wp));
            return 0;
    }
    return DefWindowProcA(hwnd, msg, wp, lp);
}

// The live state of one key, which is what a held key is. GetAsyncKeyState reads the PHYSICAL keyboard rather than
// this process's queue, so the foreground check is what keeps a key pressed in another window out of this one's
// camera - the Direct3D viewer's rule, kept word for word.
bool win_key_held(int key) {
    if (!g_hwnd || GetForegroundWindow() != g_hwnd) return false;
    return (GetAsyncKeyState(app_key_to_win32(key)) & 0x8000) != 0;
}

bool win_open(int w, int h, const char* title) {
    WNDCLASSEXA wc{}; wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW; wc.lpfnWndProc = wnd_proc; wc.hInstance = GetModuleHandleA(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW); wc.lpszClassName = "nntc_view_vk_wc";
    RegisterClassExA(&wc);
    g_hinstance = wc.hInstance;
    RECT r{ 0, 0, w, h };
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    // No WS_VISIBLE: win_show puts it on the screen, once everything that could still refuse has not.
    g_hwnd = CreateWindowA(wc.lpszClassName, title, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                           r.right - r.left, r.bottom - r.top, nullptr, nullptr, wc.hInstance, nullptr);
    if (!g_hwnd) { fprintf(stderr, "ERROR: window creation failed\n"); return false; }
    return true;
}

void win_show(void) {
    if (g_hwnd) { ShowWindow(g_hwnd, SW_SHOWNORMAL); SetForegroundWindow(g_hwnd); }
}

void win_pump(void) {
    MSG msg;
    while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageA(&msg); }
}

bool win_should_quit(void) { return g_quit || g_quit_requested; }

// The size the window ACTUALLY got, not the size that was asked for: Windows clamps a window to the work area, so on a
// smaller desktop the client rect is not the 2560x1440 of AdjustWindowRect. The swapchain is created from this, which
// is what keeps the first WM_SIZE from changing anything.
void win_client_size(int* w, int* h) {
    RECT cr{};
    if (g_hwnd && GetClientRect(g_hwnd, &cr) && cr.right > cr.left && cr.bottom > cr.top) {
        *w = (int)(cr.right - cr.left); *h = (int)(cr.bottom - cr.top);
    } else { *w = 0; *h = 0; }
}

VkSurfaceKHR win_create_surface(VkInstance instance) {
    VkWin32SurfaceCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    ci.hinstance = g_hinstance; ci.hwnd = g_hwnd;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VK_CHECK(vkCreateWin32SurfaceKHR(instance, &ci, nullptr, &surface));
    return surface;
}

double win_seconds(void) {
    LARGE_INTEGER f, t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}

// WaitMessage returns as soon as anything is queued - and a restore or a resize posts WM_SIZE, a close posts WM_CLOSE -
// so the loop wakes on exactly the events that could give it something to draw into, and on nothing else.
void win_idle(void) { WaitMessage(); }


#endif
