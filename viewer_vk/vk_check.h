// nntc_view_vk: VK_CHECK, the one error path for the Vulkan api, and the VkResult names it prints. In a header of
// its own because the platform files create the surface through it as well as main.cpp creating everything else.
#pragma once
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Every Vulkan call goes through this. A viewer has no way to carry on after a failed vkCreate* - the next call would
// be made with a null handle - so the macro says which call failed and what it returned, and stops. It is deliberately
// the only error path for the api: one shape of message, one exit code, and no call left unchecked.
// ---------------------------------------------------------------------------
static inline const char* vk_result_name(VkResult r) {   // inline: a platform file that never fails a call still includes this
    switch (r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
        default: return "an unnamed VkResult";
    }
}
#define VK_CHECK(call)                                                                                   \
    do {                                                                                                 \
        const VkResult vk_check_r = (call);                                                              \
        if (vk_check_r != VK_SUCCESS) {                                                                  \
            fprintf(stderr, "ERROR: %s returned %s (%d)\n", #call, vk_result_name(vk_check_r),           \
                    (int)vk_check_r);                                                                    \
            exit(1);                                                                                     \
        }                                                                                                \
    } while (0)
