cmake_minimum_required(VERSION 3.20)
project(nntc LANGUAGES CXX)

# The encoder is CUDA only: the host does I/O, the layout, the small dense solves and the report, and every per-texel pass
# is a kernel. It builds on Windows (MSVC + nvcc) and on Linux (gcc / clang + nvcc). The viewer is Direct3D 11 and is
# therefore built on Windows only.
#
# Configure with the Visual Studio generator and a CUDA toolset new enough for the target architectures, for example:
#   cmake -B build -S . -G "Visual Studio 18 2026" -T cuda=13.4
#   cmake --build build --config Release

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The rule for what this file says: a configure that builds everything prints STATUS lines and nothing else. Anything
# that leaves a program out or a capability compiled out - no CUDA, no Vulkan, no shaderc, no GLFW, headers too old
# for cooperative vectors - is a WARNING, which CMake prints in colour with its call site, so that a build that came
# out smaller than the tree is never a surprise found at run time. What cannot be built at all is a FATAL_ERROR.

# A single-configuration generator (Makefiles, Ninja: every Linux configure) has NO default build type, and an empty
# one is an unoptimised build that also turns the viewer's validation layer on. A developer who types the plain
# command gets Release, which is what the plain command should mean; -DCMAKE_BUILD_TYPE=Debug still asks for Debug.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Release unless asked for otherwise" FORCE)
    message(STATUS "No CMAKE_BUILD_TYPE was given: building Release")
endif()

if(WIN32)
    # On Windows the narrow C and C++ APIs - main's argv, fopen, everything a path goes through - read the process's
    # active code page, so a file named cafe with an accent could not be opened from the command line. The manifest
    # declares that code page to be UTF-8 (honoured by Windows 10 1903 and later), and MSVC embeds a .manifest listed
    # among a target's sources. Every executable gets it.
    set(NNTC_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/nntc.manifest)
endif()

# ---------------------------------------------------------------------------------------------------------------------
# The encoder, built only where a CUDA compiler is found. CUDA used to be a language of the project itself, which
# made a machine without the toolkit - an Arm laptop, an AMD or Intel box - fail at configure time before it reached
# the viewers, the very programs such a machine exists to test. Now the language is probed: with nvcc the encoder is
# built exactly as before, and without it one status line says so and the rest builds alone (on Linux that is the
# Vulkan viewer; the Direct3D viewer and bc_check are Windows programs). The
# assets those machines view are encoded elsewhere; nothing in the encoder or the format changes for this.
# ---------------------------------------------------------------------------------------------------------------------
# The probe cannot tell "this machine has no CUDA" from "the CUDA this command line asked for does not work here":
# a toolset the generator has no integration for (-T cuda=12.4 under Visual Studio 2026) or a mistyped one comes
# back NOTFOUND exactly like an Arm laptop does. So when CUDA was asked for by name, a miss is the error it always
# was, with the toolset in the message, and the silent skip is reserved for a command line that never mentioned CUDA.
# The probe's answer is also CACHED, NOTFOUND included, so a tree first configured before the toolkit went on the
# PATH keeps skipping the encoder until the cache entry is cleared; the status line says how.
include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    if(CMAKE_GENERATOR_TOOLSET MATCHES "cuda=")
        message(FATAL_ERROR "The CUDA toolset this configure asked for (-T ${CMAKE_GENERATOR_TOOLSET}) was not found or "
                            "does not work with this generator; the encoder needs a CUDA 12.8+ toolkit the generator "
                            "has integration for (leave -T out to build the viewers without the encoder)")
    endif()
    if(WIN32)
        set(NNTC_WITHOUT_CUDA "nntc_view, bc_check and nntc_view_vk still build")
    else()
        set(NNTC_WITHOUT_CUDA "nntc_view_vk still builds; nntc_view and bc_check are Windows programs")
    endif()
    message(WARNING "No CUDA compiler was found: the encoder nntc_encode is skipped (${NNTC_WITHOUT_CUDA}). "
                   "To build it after installing CUDA, re-configure with -UCMAKE_CUDA_COMPILER")
else()
enable_language(CUDA)

# sm_120 needs CUDA 12.8 or newer; the compiled floor is compute capability 8.0.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
    message(FATAL_ERROR "CUDA ${CMAKE_CUDA_COMPILER_VERSION} cannot target sm_120; configure with a CUDA 12.8+ toolset")
endif()
find_package(CUDAToolkit 12.8 REQUIRED)

add_executable(nntc_encode
    src/main.cpp
    src/mips.cpp
    src/export.cpp
    src/init.cu
    src/solve_decoder.cu
    src/solve_level1.cu
    src/solve_level0.cu
    src/refine_bc.cu
    src/objective.cu
    src/options.h
    src/image.h
    src/model.h
    src/sample.cuh
    src/device.cuh
    # shared/ is what the encoder and both viewers all read: the BC4 / BC5 pack, and the material JSON's reader, which
    # is the same two headers the viewers read the ASSET's JSON with. It started in viewer/, moved to src/ when the
    # encoder gained a reason to parse JSON of its own, and became a directory of its own when the second viewer made
    # "the encoder's sources" the wrong place for code three programs share.
    shared/bc_pack.h
    shared/nntc_json.h
    shared/json.h
    # The vendored single-header libraries, listed so that an IDE shows what the build actually compiles. All three or
    # none: the resizer became load-bearing when the source chain gained its filters, and a list that named two of the
    # three would read as a statement about the third.
    src/stb_image.h
    src/stb_image_write.h
    src/stb_image_resize2.h)

target_include_directories(nntc_encode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/shared)
target_link_libraries(nntc_encode PRIVATE CUDA::cudart_static)

# fp32 with the default FMA contraction and deliberately no --use_fast_math: the kernels are expected to agree with a
# reference evaluation to float rounding.
set_target_properties(nntc_encode PROPERTIES
    CUDA_ARCHITECTURES "80;86;89;90;120"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF)

if(MSVC)
    # CUDA::cudart_static is built against the RELEASE static C runtime in both configurations, so the encoder uses that
    # one in both too and the linker has a single runtime to resolve. Asking for the debug runtime in Debug leaves
    # LIBCMT and LIBCMTD in the same link and the linker says so. Nothing that matters in a Debug build is lost: NDEBUG
    # is still undefined, so every assert is live, and /Od /RTC1 are compiler settings and are unaffected.
    set_property(TARGET nntc_encode PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
    target_compile_definitions(nntc_encode PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(nntc_encode PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
else()
    # -ffp-contract=off on the HOST side. model.h's level1_value is compiled for the host and the device both, and its
    # multiply and add are deliberately rounded separately, because the writer's round trip and the device's snap have
    # to land on the same bits; a host compiler that fused them into one multiply-add (gcc contracts by default, and
    # -march=native makes it likely) would move the last bit of a value, and verify_level1_on_grid would then refuse to
    # write an asset that is correct. MSVC does not contract under its default /fp:precise, so its branch needs nothing.
    target_compile_options(nntc_encode PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall$<SEMICOLON>-Wextra$<SEMICOLON>-ffp-contract=off>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall$<SEMICOLON>-Xcompiler=-Wextra$<SEMICOLON>-Xcompiler=-ffp-contract=off>)
endif()
if(WIN32)
    target_sources(nntc_encode PRIVATE ${NNTC_MANIFEST})
endif()
endif()   # the CUDA compiler was found

# ---------------------------------------------------------------------------------------------------------------------
# The viewer: Direct3D 11, Windows only, plain C++17 with no dependency beyond the C runtime, Win32, d3d11 and d3dcompiler.
# ---------------------------------------------------------------------------------------------------------------------
if(WIN32)
    # shared/bc_pack.h is the one BC4 / BC5 encoder in the tree: the encoder packs level 0 with it and the viewer packs
    # an uncompressed level 0 with it at load, so the two produce the same bytes by construction.
    add_executable(nntc_view viewer/main.cpp shared/bc_pack.h shared/nntc_json.h shared/json.h shared/dds.h)
    target_sources(nntc_view PRIVATE ${NNTC_MANIFEST})
    target_include_directories(nntc_view PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shared)
    target_link_libraries(nntc_view PRIVATE d3d11 d3dcompiler)

    # bc_check packs a .dds the way the viewer packs level 0 and decodes it with iOrange's bcdec (MIT / Unlicense).
    add_executable(bc_check viewer/bc_check.cpp shared/bc_pack.h shared/nntc_json.h shared/json.h viewer/bcdec.h)
    target_sources(bc_check PRIVATE ${NNTC_MANIFEST})
    target_include_directories(bc_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shared)

    # The same warning level as the encoder. shared/json.h and viewer/bcdec.h are vendored and are included with the
    # warnings turned down where they are used, so /W4 here covers the code this tree owns.
    if(MSVC)
        target_compile_definitions(nntc_view PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_definitions(bc_check PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(nntc_view PRIVATE /W4)
        target_compile_options(bc_check PRIVATE /W4)
    else()
        target_compile_options(nntc_view PRIVATE -Wall -Wextra)
        target_compile_options(bc_check PRIVATE -Wall -Wextra)
    endif()

    # The viewer compiles nntc_view.hlsl at runtime from the working directory or from beside the executable, so the
    # shader travels with the build (edit it and press R to reload).
    add_custom_command(TARGET nntc_view POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/viewer/bin/nntc_view.hlsl"
                "$<TARGET_FILE_DIR:nntc_view>/nntc_view.hlsl")
endif()

# ---------------------------------------------------------------------------------------------------------------------
# The second viewer: Vulkan 1.3, the same asset through another api (docs/VULKAN_VIEWER_PLAN.md). It is built wherever
# the Vulkan SDK is found - Windows today, Linux when the box arrives - and a tree without an SDK still configures and
# still builds everything else, which is why the whole block is guarded and says so rather than failing.
#
# The loader is linked normally through CMake's own FindVulkan: no volk, no memory allocator, no windowing library and
# nothing vendored. shaderc_combined comes from the same SDK and is what turns the two GLSL files into SPIR-V at
# runtime, so a shader can be edited beside the executable exactly as the other viewer's .hlsl can; it is a large
# static library and that is accepted, because this is a sample. Without it there is no way to compile a shader at all,
# so the target is skipped rather than half-built.
# ---------------------------------------------------------------------------------------------------------------------
find_package(Vulkan QUIET OPTIONAL_COMPONENTS shaderc_combined)

# Which shaderc. On Windows it is the SDK's shaderc_combined, one complete static archive, and the shipped .exe
# depends on vulkan-1.dll alone. On Linux a distribution's libshaderc-dev is the usual source, and its
# libshaderc_combined.a is NOT complete on every release: newer packages build shaderc against the system's own
# SPIRV-Tools and glslang and leave them out of the archive, so a link against it alone fails with undefined
# spvValidatorOptionsDestroy and spvtools::Optimizer symbols - on a Kubuntu box, and on Ubuntu 24.04's own
# libshaderc-dev 2023.8, whose libshaderc_combined.a is 242 KB and defines neither of those two symbols. The
# distribution's SHARED libshaderc carries its dependencies as its own NEEDED entries and links
# on every release, so on Linux it is preferred and the combined archive is the fallback for a LunarG SDK install
# that has no shared library on the path. The header is the same shaderc/shaderc.h either way and the viewer's
# source does not know which it got.
set(NNTC_SHADERC "")
if(NOT WIN32)
    find_library(NNTC_SHADERC_SHARED_LIBRARY NAMES shaderc_shared shaderc)
    find_path(NNTC_SHADERC_INCLUDE_DIR shaderc/shaderc.h)
    if(NNTC_SHADERC_SHARED_LIBRARY AND NNTC_SHADERC_INCLUDE_DIR)
        add_library(nntc_shaderc_shared SHARED IMPORTED)
        set_target_properties(nntc_shaderc_shared PROPERTIES
            IMPORTED_LOCATION "${NNTC_SHADERC_SHARED_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NNTC_SHADERC_INCLUDE_DIR}")
        set(NNTC_SHADERC nntc_shaderc_shared)
        set(NNTC_SHADERC_WHICH "the shared libshaderc at ${NNTC_SHADERC_SHARED_LIBRARY}")
    endif()
endif()
if(NOT NNTC_SHADERC AND TARGET Vulkan::shaderc_combined)
    set(NNTC_SHADERC Vulkan::shaderc_combined)
    set(NNTC_SHADERC_WHICH "the SDK's shaderc_combined")
endif()

if(NOT Vulkan_FOUND)
    message(WARNING "Vulkan was not found: the target nntc_view_vk is skipped (everything else still builds)")
elseif(NOT NNTC_SHADERC)
    message(WARNING "No shaderc was found (neither the SDK's shaderc_combined nor a shared libshaderc with its header): "
                   "nntc_view_vk is skipped (it compiles its shaders at runtime)")
else()
    message(STATUS "nntc_view_vk compiles its shaders with ${NNTC_SHADERC_WHICH}")

    # WHICH VULKAN HEADERS, said at configure time. viewer_vk/main.cpp needs 204 or newer (Vulkan 1.3 core: dynamic
    # rendering and synchronisation2) and compiles its cooperative-vector path in only where the headers carry
    # VK_NV_cooperative_vector, which first appears in Vulkan-Headers 1.4.307. A distribution's stock headers are
    # routinely older - Ubuntu 22.04 ships 204, Debian 12 ships 239, Ubuntu 24.04 ships 275 - so without this line a
    # user whose accelerated path is simply absent would have nothing to read but the viewer's own run-time refusal.
    # The number is read out of the header rather than guessed, because FindVulkan reports no version on every
    # platform. (FindVulkan already honours VULKAN_SDK before the system paths, so a LunarG install on PATH wins by
    # itself and needs nothing here; -DVulkan_INCLUDE_DIR=... points at one that is not.)
    set(NNTC_VK_HEADER_VERSION "")
    if(EXISTS "${Vulkan_INCLUDE_DIR}/vulkan/vulkan_core.h")
        file(STRINGS "${Vulkan_INCLUDE_DIR}/vulkan/vulkan_core.h" NNTC_VK_HEADER_LINE
             REGEX "^#define[ \t]+VK_HEADER_VERSION[ \t]+[0-9]+")
        if(NNTC_VK_HEADER_LINE)
            string(REGEX REPLACE "^#define[ \t]+VK_HEADER_VERSION[ \t]+([0-9]+).*$" "\\1"
                   NNTC_VK_HEADER_VERSION "${NNTC_VK_HEADER_LINE}")
        endif()
    endif()
    if(NOT NNTC_VK_HEADER_VERSION)
        message(WARNING "nntc_view_vk: the Vulkan header version could not be read from ${Vulkan_INCLUDE_DIR}; "
                       "the build itself still refuses anything older than 204")
    elseif(NNTC_VK_HEADER_VERSION LESS 307)
        message(WARNING "nntc_view_vk: Vulkan headers 1.x.${NNTC_VK_HEADER_VERSION} in ${Vulkan_INCLUDE_DIR} - "
                        "cooperative vectors are compiled OUT (VK_NV_cooperative_vector needs 1.4.307 or newer). "
                        "The plain decode path, which is the product, is what runs; on AMD and Intel nothing is "
                        "missing, since the extension is NVIDIA's alone")
    else()
        message(STATUS "nntc_view_vk: Vulkan headers 1.x.${NNTC_VK_HEADER_VERSION} in ${Vulkan_INCLUDE_DIR} - "
                       "cooperative vectors are compiled IN; whether they RUN is still the device's own query")
    endif()
    add_executable(nntc_view_vk
        viewer_vk/main.cpp viewer_vk/platform_win32.cpp viewer_vk/platform_linux.cpp
        viewer_vk/platform.h viewer_vk/vk_check.h
        shared/bc_pack.h shared/nntc_json.h shared/json.h shared/dds.h)
    target_include_directories(nntc_view_vk PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shared)
    # shaderc/shaderc.h, where find_path found it. The imported shared library carries this itself, so this line
    # is for the combined archive: Vulkan::shaderc_combined publishes the SDK's own include directory, which is
    # the right one for an SDK install and is not the right one for a distribution whose loader headers and whose
    # shaderc header sit in different prefixes. Naming the directory the library was found beside costs a
    # duplicate -I in the usual case and is the difference between building and not in the other.
    if(NNTC_SHADERC_INCLUDE_DIR)
        target_include_directories(nntc_view_vk PRIVATE ${NNTC_SHADERC_INCLUDE_DIR})
    endif()
    target_link_libraries(nntc_view_vk PRIVATE Vulkan::Vulkan ${NNTC_SHADERC})
    if(WIN32)
        target_sources(nntc_view_vk PRIVATE ${NNTC_MANIFEST})
        target_compile_definitions(nntc_view_vk PRIVATE VK_USE_PLATFORM_WIN32_KHR _CRT_SECURE_NO_WARNINGS)
        if(MSVC)
            target_compile_options(nntc_view_vk PRIVATE /W4)
            # LNK4099, and only LNK4099. The SDK ships shaderc_combinedd.lib without the .pdb files of the libraries
            # it was built from, so a Debug link prints one "PDB was not found" line per object of a vendored static
            # library - several hundred of them, which bury anything this tree's own code has to say. It is a statement
            # about the SDK's packaging and not about this program, and no other diagnostic is silenced.
            target_link_options(nntc_view_vk PRIVATE /IGNORE:4099)
        endif()
    else()
        # The same warning level as the encoder and the other viewer: the source is clean under it, and the build
        # is what keeps it so.
        target_compile_options(nntc_view_vk PRIVATE -Wall -Wextra)
        # GLFW is the one windowing dependency, and it is optional: without it the viewer still builds and its
        # --shot - which creates no window and no surface at all - is what runs on such a machine. GLFW 3 is the
        # small C library every Vulkan sample uses for its window. A GLFW 3.4 (Fedora, Arch, Debian 13) speaks
        # Wayland and X11 both and chooses at run time; a 3.3 (Ubuntu 22.04 and 24.04, Debian 12, Mint, Pop) is
        # built for one of them, X11 unless the distribution's libglfw3-wayland is installed instead, and runs on a
        # Wayland desktop through Xwayland. Either way the window works. The distributions package it as
        # libglfw3-dev, glfw-devel, libglfw-devel or glfw; its CMake config file is looked for first and its
        # pkg-config file second.
        find_package(glfw3 3.3 CONFIG QUIET)
        if(TARGET glfw)
            set(NNTC_GLFW_TARGET glfw)
        else()
            find_package(PkgConfig QUIET)
            if(PkgConfig_FOUND)
                pkg_check_modules(GLFW3 QUIET IMPORTED_TARGET glfw3)
            endif()
            if(TARGET PkgConfig::GLFW3)
                set(NNTC_GLFW_TARGET PkgConfig::GLFW3)
            endif()
        endif()
        if(NNTC_GLFW_TARGET)
            target_compile_definitions(nntc_view_vk PRIVATE NNTC_GLFW)
            target_link_libraries(nntc_view_vk PRIVATE ${NNTC_GLFW_TARGET})
            message(STATUS "nntc_view_vk: the window is GLFW's (${NNTC_GLFW_TARGET})")
        else()
            message(WARNING "GLFW was not found: nntc_view_vk builds HEADLESS, so --shot runs and the window does "
                            "not. Install the development package (libglfw3-dev on Debian and Ubuntu, glfw-devel "
                            "on Fedora, libglfw-devel on openSUSE, glfw on Arch) and run cmake again")
        endif()
    endif()

    # The two GLSL files travel with the build and are compiled at runtime from BESIDE THE EXECUTABLE and from nowhere
    # else: edit the copy this command writes and the next run, or the key R, picks it up. The viewer used to look in
    # the working directory first, where a stray view.frag silently replaced the real one, so this copy is now the
    # only file it can read.
    foreach(NNTC_VK_SHADER view.vert view.frag view_coopvec.frag)
        add_custom_command(TARGET nntc_view_vk POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_CURRENT_SOURCE_DIR}/viewer_vk/bin/${NNTC_VK_SHADER}"
                    "$<TARGET_FILE_DIR:nntc_view_vk>/${NNTC_VK_SHADER}")
    endforeach()
endif()

# ---------------------------------------------------------------------------------------------------------------------
# Install: every program that was built, and beside each viewer the shader files it reads from beside itself, so that
# `cmake --install build --prefix DIR` gives a DIR/bin that runs from anywhere; the build tree runs as it is too.
# ---------------------------------------------------------------------------------------------------------------------
foreach(NNTC_PROGRAM nntc_encode nntc_view bc_check nntc_view_vk)
    if(TARGET ${NNTC_PROGRAM})
        install(TARGETS ${NNTC_PROGRAM} RUNTIME DESTINATION bin)
    endif()
endforeach()
if(TARGET nntc_view)
    install(FILES viewer/bin/nntc_view.hlsl DESTINATION bin)
endif()
if(TARGET nntc_view_vk)
    install(FILES viewer_vk/bin/view.vert viewer_vk/bin/view.frag viewer_vk/bin/view_coopvec.frag DESTINATION bin)
endif()
