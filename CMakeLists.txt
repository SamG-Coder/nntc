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
# The encoder. It is ALWAYS built; what a CUDA compiler decides is which BACKEND it carries, not whether the program
# exists. CUDA used to be a language of the project itself, which made a machine without the toolkit - an Arm laptop,
# an AMD or Intel box - fail at configure time before it reached the viewers, the very programs such a machine exists
# to test. Then the language was probed and the encoder was skipped without it. Now the encoder's device interface has
# two arms behind a dispatcher (docs/CPU_BACKEND_PLAN.md section 1.7), so a machine with no toolkit builds the encoder
# with the CPU arm alone and a machine with one builds it exactly as before.
#
# That CPU-only build is not only for those machines. It is the proof that the dispatch is complete: without nvcc the
# six .cu files are not compiled, the global symbols they define do not exist, and a call site that still named one
# would fail to LINK with the linker saying which. A CPU-only build that links clean is a machine-checked statement
# about the whole host side.
# ---------------------------------------------------------------------------------------------------------------------
# The probe cannot tell "this machine has no CUDA" from "the CUDA this command line asked for does not work here":
# a toolset the generator has no integration for (-T cuda=12.4 under Visual Studio 2026) or a mistyped one comes
# back NOTFOUND exactly like an Arm laptop does. So when CUDA was asked for by name, a miss is the error it always
# was, with the toolset in the message, and the silent skip is reserved for a command line that never mentioned CUDA.
# The probe's answer is also CACHED, NOTFOUND included, so a tree first configured before the toolkit went on the
# PATH keeps building without the CUDA backend until the cache entry is cleared; the warning says how.

# The host sources, which every configure compiles. backend.cpp routes each of the seam's 46 calls to an arm and names
# no arm's internals; cpu/ is the CPU arm, which is built everywhere because that is what makes a link with no CUDA in
# it possible at all.
set(NNTC_HOST_SOURCES
    src/main.cpp
    src/mips.cpp
    src/export.cpp
    src/backend.cpp
    src/backend.h
    src/cpu/cpu_backend.cpp
    src/cpu/cpu_backend.h
    src/cpu/cpu_model.cpp
    src/cpu/cpu_model.h
    src/cpu/cpu_objective.cpp
    src/cpu/cpu_init.cpp
    src/cpu/cpu_solve_decoder.cpp
    src/cpu/cpu_solve_level1.cpp
    src/cpu/cpu_solve_level0.cpp
    src/cpu/cpu_solve_level0.h
    src/cpu/cpu_refine_bc.cpp
    src/cpu/cpu_refine_bc.h
    src/cpu/cpu_constants.h
    src/cpu/cpu_sample.h
    src/cpu/pool.h
    src/options.h
    src/image.h
    src/model.h
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

# The CUDA arm: the six kernel files, the two headers they own, and the one host file that names what they define. It
# is one list per arm on purpose - a third arm is a third list and a third probe, and this one does not move. The
# per-kernel harness (--backend check) rides with it: it compares the two arms, so it exists only where both do. Its
# probe (backend_check_probe.cu) is compiled by nvcc because it reads the device model's buffers, but it is not a
# kernel file: it launches nothing and writes nothing, and the harness alone calls it.
set(NNTC_CUDA_SOURCES
    src/init.cu
    src/solve_decoder.cu
    src/solve_level1.cu
    src/solve_level0.cu
    src/refine_bc.cu
    src/objective.cu
    src/sample.cuh
    src/device.cuh
    src/cuda_backend.cpp
    src/cuda_backend.h
    src/backend_check.cpp
    src/backend_check.h
    src/backend_check_probe.cu
    src/backend_check_probe.h)

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    if(CMAKE_GENERATOR_TOOLSET MATCHES "cuda=")
        message(FATAL_ERROR "The CUDA toolset this configure asked for (-T ${CMAKE_GENERATOR_TOOLSET}) was not found or "
                            "does not work with this generator; the encoder's CUDA backend needs a CUDA 12.8+ toolkit "
                            "the generator has integration for (leave -T out to build without that backend)")
    endif()
    if(WIN32)
        set(NNTC_WITHOUT_CUDA "nntc_view, bc_check and nntc_view_vk still build")
    else()
        set(NNTC_WITHOUT_CUDA "nntc_view_vk still builds; nntc_view and bc_check are Windows programs")
    endif()
    message(WARNING "No CUDA compiler was found: the encoder nntc_encode is built with its CPU backend only. It "
                    "encodes, with the same algorithm, several times slower than on an NVIDIA GPU; --backend cuda "
                    "refuses by name. (${NNTC_WITHOUT_CUDA}.) "
                    "The CUDA backend needs the CUDA toolkit, 12.8 or newer: on Windows the CUDA installer with its "
                    "Visual Studio integration, then -T cuda=13.4 (or your version) on the cmake line; on Linux the "
                    "cuda-toolkit-13-x package from NVIDIA's repository (the distributions' own nvidia-cuda-toolkit "
                    "is usually too old) with /usr/local/cuda/bin on the PATH. Then re-configure with "
                    "-UCMAKE_CUDA_COMPILER, since the miss is cached")
else()
    enable_language(CUDA)

    # sm_120 needs CUDA 12.8 or newer; the compiled floor is compute capability 8.0.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
        message(FATAL_ERROR "CUDA ${CMAKE_CUDA_COMPILER_VERSION} cannot target sm_120; configure with a CUDA 12.8+ "
                            "toolset")
    endif()
    find_package(CUDAToolkit 12.8 REQUIRED)
endif()

add_executable(nntc_encode ${NNTC_HOST_SOURCES})

target_include_directories(nntc_encode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/shared)
# The CPU arm is std::thread's, wherever it is built, which is everywhere.
find_package(Threads REQUIRED)
target_link_libraries(nntc_encode PRIVATE Threads::Threads)

if(CMAKE_CUDA_COMPILER)
    # NNTC_CUDA is what compiles the dispatcher's CUDA branch in. It is defined in the same block that adds the .cu
    # files, so the define and the kernels can never disagree: no nvcc, no kernels, no branch, and then the linker has
    # something to say about any call site that was missed.
    target_sources(nntc_encode PRIVATE ${NNTC_CUDA_SOURCES})
    target_compile_definitions(nntc_encode PRIVATE NNTC_CUDA)
    target_link_libraries(nntc_encode PRIVATE CUDA::cudart_static)

    # Deliberately no --use_fast_math: the kernels are expected to agree with a reference evaluation to float rounding.
    set_target_properties(nntc_encode PROPERTIES
        CUDA_ARCHITECTURES "80;86;89;90;120"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF)
    # Fused multiply-add. nvcc fuses a * b + c into one fma by default and the C++ compilers here do not (MSVC's
    # /fp:precise, and -ffp-contract=off on gcc and clang), so with contraction on, the CUDA and CPU backends round the
    # same expression differently. OFF, the default, compiles the kernels with -fmad=false so both backends round
    # every multiply and every add separately, which keeps the two as close as the arithmetic allows. ON restores
    # nvcc's default, which is what every CUDA asset, test and tracked log before this option was made with: build
    # with -DNNTC_CUDA_FMAD=ON to reproduce them byte for byte. The kernel sources are the same either way; only how
    # nvcc compiles them changes. The two expressions that must agree host-versus-device (model.h's level1_value and
    # refine_bc.cu's bc0_value) use __dadd_rn / __dmul_rn explicitly and are unaffected by this option.
    option(NNTC_CUDA_FMAD "compile the CUDA kernels with nvcc's fused multiply-add contraction (the pre-option default)" OFF)
    if(NOT NNTC_CUDA_FMAD)
        target_compile_options(nntc_encode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>)
    endif()
    message(STATUS "nntc_encode: CUDA fused multiply-add ${NNTC_CUDA_FMAD} (NNTC_CUDA_FMAD; ON reproduces the pre-option CUDA assets)")
endif()

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

# The unit exercise of the CPU backend (tests/cpu_check.cpp): the thread pool's contract and the model's transfers
# proved on their own, before any kernel rests on them, and then every ported kernel run at several thread counts and
# held to the same bits at each - from the CPU arm's sources alone, so it builds on every configure, and it is the
# program a ThreadSanitizer build runs threaded (docs/CPU_BACKEND_PLAN.md section 3.4). `nntc_cpu_check --bench`
# takes stage C1's two measurements. The BC refinement (stage C7) packs through export.cpp's host code as the CUDA arm
# does, and export.cpp reaches decode_plane through the dispatcher, so the dispatcher and the rest of the CPU arm come
# too - compiled without NNTC_CUDA, as in the CPU-only configure, so nothing here needs a CUDA toolkit.
add_executable(nntc_cpu_check tests/cpu_check.cpp src/cpu/cpu_model.cpp src/cpu/cpu_model.h src/cpu/pool.h
               src/cpu/cpu_constants.h src/cpu/cpu_backend.h src/cpu/cpu_sample.h src/cpu/cpu_objective.cpp
               src/cpu/cpu_init.cpp src/cpu/cpu_solve_decoder.cpp src/cpu/cpu_solve_level1.cpp
               src/cpu/cpu_solve_level0.cpp src/cpu/cpu_solve_level0.h src/cpu/cpu_refine_bc.cpp
               src/cpu/cpu_refine_bc.h src/cpu/cpu_backend.cpp src/backend.cpp src/backend.h src/export.cpp)
target_include_directories(nntc_cpu_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/shared)
target_link_libraries(nntc_cpu_check PRIVATE Threads::Threads)
if(MSVC)
    set_property(TARGET nntc_cpu_check PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
    target_compile_definitions(nntc_cpu_check PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(nntc_cpu_check PRIVATE /W4)
else()
    target_compile_options(nntc_cpu_check PRIVATE -Wall -Wextra -ffp-contract=off)
endif()
if(WIN32)
    target_sources(nntc_cpu_check PRIVATE ${NNTC_MANIFEST})
endif()

# Which backends this encoder came out with, on one line, so that a configure log answers the question without anyone
# having to read the warnings above it for an absence. The CPU arm is always there; the CUDA one is the variable.
if(CMAKE_CUDA_COMPILER)
    set(NNTC_HAS_CUDA_TEXT " and cuda")
else()
    set(NNTC_HAS_CUDA_TEXT "")
endif()
message(STATUS "nntc_encode backends: cpu${NNTC_HAS_CUDA_TEXT}")

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
    # shader travels with the build: edit viewer/bin/nntc_view.hlsl, build, and press R. An edit made to the copy beside
    # the executable lasts only until the next build overwrites it. The copy is its own target, run on every build
    # (copy_if_different makes an unchanged shader a no-op), rather than a post-build step: with a Makefile or Ninja
    # generator a post-build step runs only when the executable relinks, so a change to the shader alone left a stale
    # copy beside it.
    add_custom_target(nntc_view_shader ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:nntc_view>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/viewer/bin/nntc_view.hlsl"
                "$<TARGET_FILE_DIR:nntc_view>/nntc_view.hlsl"
        VERBATIM)
    add_dependencies(nntc_view nntc_view_shader)

    # The third viewer: the same asset on Direct3D 12 (docs/D3D12_LINEAR_ALGEBRA_PLAN.md part I). It compiles the SAME
    # viewer/bin/nntc_view.hlsl to the same vs_5_0 / ps_5_0 DXBC through the same d3dcompiler, so the two Direct3D
    # viewers hand their drivers the same bytes and any difference between their frames is in the api path, which is the
    # thing it exists to check. Nothing preview is involved: d3d12 and dxgi are the Windows SDK's, there is no Agility
    # SDK and no shader model past 5.0, so it builds and runs wherever nntc_view does.
    add_executable(nntc_view_d3d12 viewer_d3d12/main.cpp shared/bc_pack.h shared/nntc_json.h shared/json.h shared/dds.h)
    target_sources(nntc_view_d3d12 PRIVATE ${NNTC_MANIFEST})
    target_include_directories(nntc_view_d3d12 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shared)
    target_link_libraries(nntc_view_d3d12 PRIVATE d3d12 dxgi d3dcompiler)
    if(MSVC)
        target_compile_definitions(nntc_view_d3d12 PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(nntc_view_d3d12 PRIVATE /W4)
    else()
        target_compile_options(nntc_view_d3d12 PRIVATE -Wall -Wextra)
    endif()

    # It reads nntc_view.hlsl from BESIDE THE EXECUTABLE and from nowhere else (the Vulkan viewer's rule: a stray shader
    # in the working directory must not be the one that is compiled), so the copy is not optional for it. With a
    # multi-config generator this is the same directory nntc_view's copy lands in and copy_if_different makes the second
    # copy a no-op; with a single-config one they can differ, which is why it is its own target rather than a shared one.
    add_custom_target(nntc_view_d3d12_shader ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:nntc_view_d3d12>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/viewer/bin/nntc_view.hlsl"
                "$<TARGET_FILE_DIR:nntc_view_d3d12>/nntc_view.hlsl"
        VERBATIM)
    add_dependencies(nntc_view_d3d12 nntc_view_d3d12_shader)

    # -----------------------------------------------------------------------------------------------------------------
    # PHASE B, and it is OFF BY DEFAULT: the optional Shader Model 6.10 linear-algebra decode path
    # (docs/D3D12_LINEAR_ALGEBRA_PLAN.md part II, viewer_d3d12/PHASE_B_NOTES.md).
    #
    # OFF, nntc_view_d3d12 is exactly what it is above: the Windows SDK's Direct3D 12, no NuGet package, no
    # D3D12SDKVersion export, nothing preview anywhere in the build, and the same frames byte for byte.
    #
    # ON, three PREVIEW NuGet packages are fetched at pinned versions into the BUILD directory - never into the source
    # tree and never onto the machine - and their binaries are copied beside the executable. The .nupkg files are
    # ordinary zip archives, so they are fetched with file(DOWNLOAD) and unpacked with file(ARCHIVE_EXTRACT) rather than
    # through nuget.exe, which would be a fourth download for nothing. A configure with no network prints a WARNING
    # naming what it wanted and builds the viewer WITHOUT the path, in this tree's style: the plain decode is the
    # product and a missing preview package must never fail a build.
    # -----------------------------------------------------------------------------------------------------------------
    option(NNTC_D3D12_LINALG "Build nntc_view_d3d12 with the optional Shader Model 6.10 linear-algebra decode path (fetches three preview NuGet packages)" OFF)
    if(NNTC_D3D12_LINALG)
        # The pins. They are Microsoft's own sample's, and the runtime specification they implement is a 0.9 draft: when
        # it reaches 1.0 these three lines and the plan's section B.1.1 are re-read against it.
        set(NNTC_LINALG_AGILITY_VERSION "1.721.3-preview")
        set(NNTC_LINALG_DXC_VERSION     "1.10.2605.37-preview")
        set(NNTC_LINALG_WARP_VERSION    "1.65535.20-preview")
        set(NNTC_LINALG_ROOT "${CMAKE_BINARY_DIR}/nntc_linalg_packages")

        # Which of the packages' architecture folders this build wants. A 32-bit build is refused by name rather than
        # half-configured: the packages do publish a win32 folder, but nothing else in this tree is built 32-bit and an
        # untested arm of an optional preview path is not worth carrying.
        set(NNTC_LINALG_ARCH "x64")
        if(CMAKE_VS_PLATFORM_NAME MATCHES "^([Aa][Rr][Mm]64)$" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^([Aa][Rr][Mm]64)$")
            set(NNTC_LINALG_ARCH "arm64")
        endif()

        set(NNTC_LINALG_OK TRUE)
        if(CMAKE_SIZEOF_VOID_P EQUAL 4)
            message(WARNING "NNTC_D3D12_LINALG is ON and this is a 32-bit build; the linear-algebra path is built for "
                            "x64 and arm64 only, so nntc_view_d3d12 is built WITHOUT it")
            set(NNTC_LINALG_OK FALSE)
        endif()
        foreach(NNTC_PKG "microsoft.direct3d.d3d12:${NNTC_LINALG_AGILITY_VERSION}"
                         "microsoft.direct3d.dxc:${NNTC_LINALG_DXC_VERSION}"
                         "microsoft.direct3d.warp:${NNTC_LINALG_WARP_VERSION}")
            string(REPLACE ":" ";" NNTC_PKG_PARTS "${NNTC_PKG}")
            list(GET NNTC_PKG_PARTS 0 NNTC_PKG_ID)
            list(GET NNTC_PKG_PARTS 1 NNTC_PKG_VER)
            set(NNTC_PKG_DIR "${NNTC_LINALG_ROOT}/${NNTC_PKG_ID}.${NNTC_PKG_VER}")
            # An unpacked package is kept: the three of them are a hundred megabytes and re-running cmake must not
            # fetch them again. Deleting the build directory is what re-fetches them.
            if(NOT EXISTS "${NNTC_PKG_DIR}/build/native")
                set(NNTC_PKG_URL "https://api.nuget.org/v3-flatcontainer/${NNTC_PKG_ID}/${NNTC_PKG_VER}/${NNTC_PKG_ID}.${NNTC_PKG_VER}.nupkg")
                message(STATUS "nntc_view_d3d12: fetching ${NNTC_PKG_ID} ${NNTC_PKG_VER}")
                file(DOWNLOAD "${NNTC_PKG_URL}" "${NNTC_LINALG_ROOT}/${NNTC_PKG_ID}.${NNTC_PKG_VER}.nupkg"
                     STATUS NNTC_PKG_STATUS INACTIVITY_TIMEOUT 60)
                list(GET NNTC_PKG_STATUS 0 NNTC_PKG_CODE)
                if(NNTC_PKG_CODE EQUAL 0)
                    file(ARCHIVE_EXTRACT INPUT "${NNTC_LINALG_ROOT}/${NNTC_PKG_ID}.${NNTC_PKG_VER}.nupkg"
                         DESTINATION "${NNTC_PKG_DIR}")
                else()
                    list(GET NNTC_PKG_STATUS 1 NNTC_PKG_WHY)
                    message(WARNING "nntc_view_d3d12: ${NNTC_PKG_URL} could not be fetched (${NNTC_PKG_WHY})")
                    set(NNTC_LINALG_OK FALSE)
                endif()
            endif()
        endforeach()

        set(NNTC_LINALG_AGILITY "${NNTC_LINALG_ROOT}/microsoft.direct3d.d3d12.${NNTC_LINALG_AGILITY_VERSION}/build/native")
        set(NNTC_LINALG_DXC "${NNTC_LINALG_ROOT}/microsoft.direct3d.dxc.${NNTC_LINALG_DXC_VERSION}/build/native")
        set(NNTC_LINALG_WARP "${NNTC_LINALG_ROOT}/microsoft.direct3d.warp.${NNTC_LINALG_WARP_VERSION}/build/native")
        foreach(NNTC_LINALG_FILE "${NNTC_LINALG_AGILITY}/include/d3d12.h"
                                 "${NNTC_LINALG_AGILITY}/bin/${NNTC_LINALG_ARCH}/D3D12Core.dll"
                                 "${NNTC_LINALG_DXC}/include/dxcapi.h"
                                 "${NNTC_LINALG_DXC}/include/hlsl/dx/linalg.h"
                                 "${NNTC_LINALG_DXC}/bin/${NNTC_LINALG_ARCH}/dxcompiler.dll"
                                 "${NNTC_LINALG_DXC}/bin/${NNTC_LINALG_ARCH}/dxil.dll"
                                 "${NNTC_LINALG_WARP}/bin/${NNTC_LINALG_ARCH}/d3d10warp.dll")
            if(NOT EXISTS "${NNTC_LINALG_FILE}")
                message(WARNING "nntc_view_d3d12: ${NNTC_LINALG_FILE} is not there")
                set(NNTC_LINALG_OK FALSE)
            endif()
        endforeach()

        if(NOT NNTC_LINALG_OK)
            message(WARNING "NNTC_D3D12_LINALG is ON and the preview packages could not be prepared, so nntc_view_d3d12 "
                            "is built WITHOUT the linear-algebra path (everything else is unaffected). It needs "
                            "Microsoft.Direct3D.D3D12 ${NNTC_LINALG_AGILITY_VERSION}, Microsoft.Direct3D.DXC "
                            "${NNTC_LINALG_DXC_VERSION} and Microsoft.Direct3D.WARP ${NNTC_LINALG_WARP_VERSION} for "
                            "${NNTC_LINALG_ARCH}, fetched from api.nuget.org into ${NNTC_LINALG_ROOT}; a machine with no "
                            "network can unpack the three .nupkg files there by hand and configure again")
        else()
            # The Agility SDK's headers go FIRST on the include path, which is what makes <d3d12.h> the preview one.
            target_include_directories(nntc_view_d3d12 BEFORE PRIVATE
                "${NNTC_LINALG_AGILITY}/include" "${NNTC_LINALG_DXC}/include")
            # dxcompiler.lib is deliberately NOT linked: main.cpp loads dxcompiler.dll by hand, so an executable copied
            # away from the files below still runs and still draws the plain path instead of failing in the loader.
            target_compile_definitions(nntc_view_d3d12 PRIVATE NNTC_D3D12_LINALG=1)
            target_sources(nntc_view_d3d12 PRIVATE viewer_d3d12/bin/nntc_view_linalg.hlsl)
            set_source_files_properties(viewer_d3d12/bin/nntc_view_linalg.hlsl PROPERTIES HEADER_FILE_ONLY TRUE)

            # What has to sit beside the executable, and WHERE, which is not all one place:
            #   D3D12\D3D12Core.dll and D3D12\d3d12SDKLayers.dll - the folder the D3D12SDKPath export names;
            #   d3d10warp.dll BESIDE THE EXECUTABLE - which is where the WARP package's own .targets file puts it, and
            #     where it has to be: a copy of it under D3D12\ is NOT picked up, and the run then gets the system
            #     WARP, which answers Shader Model 6.8 and linear algebra tier 0 (measured; the plan expected the
            #     D3D12\ folder and was wrong about this one file);
            #   dxcompiler.dll and dxil.dll - the preview compiler and its signing library;
            #   dx\linalg.h and nntc_view_linalg.hlsl - what that compiler reads at run time.
            add_custom_target(nntc_view_d3d12_linalg ALL
                COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:nntc_view_d3d12>/D3D12"
                COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:nntc_view_d3d12>/dx"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${NNTC_LINALG_AGILITY}/bin/${NNTC_LINALG_ARCH}/D3D12Core.dll"
                        "$<TARGET_FILE_DIR:nntc_view_d3d12>/D3D12/D3D12Core.dll"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${NNTC_LINALG_AGILITY}/bin/${NNTC_LINALG_ARCH}/d3d12SDKLayers.dll"
                        "$<TARGET_FILE_DIR:nntc_view_d3d12>/D3D12/d3d12SDKLayers.dll"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${NNTC_LINALG_WARP}/bin/${NNTC_LINALG_ARCH}/d3d10warp.dll"
                        "$<TARGET_FILE_DIR:nntc_view_d3d12>/d3d10warp.dll"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${NNTC_LINALG_DXC}/bin/${NNTC_LINALG_ARCH}/dxcompiler.dll"
                        "$<TARGET_FILE_DIR:nntc_view_d3d12>/dxcompiler.dll"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${NNTC_LINALG_DXC}/bin/${NNTC_LINALG_ARCH}/dxil.dll"
                        "$<TARGET_FILE_DIR:nntc_view_d3d12>/dxil.dll"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${NNTC_LINALG_DXC}/include/hlsl/dx/linalg.h"
                        "$<TARGET_FILE_DIR:nntc_view_d3d12>/dx/linalg.h"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${CMAKE_CURRENT_SOURCE_DIR}/viewer_d3d12/bin/nntc_view_linalg.hlsl"
                        "$<TARGET_FILE_DIR:nntc_view_d3d12>/nntc_view_linalg.hlsl"
                VERBATIM)
            add_dependencies(nntc_view_d3d12 nntc_view_d3d12_linalg)
            # The same files again, for the install rules at the end of this file. They are named here, inside the
            # branch where the packages are known to be present, and that is what makes those rules inert everywhere
            # else: unset variables, so the install block's own test fails. Shipping them is not optional packaging
            # in any case: the executable's D3D12SDKVersion export names this
            # runtime, so an installed copy without D3D12\D3D12Core.dll beside it gets NO DEVICE AT ALL - not the
            # plain path, no device - which is exactly the trap PHASE_B_NOTES.md warns a reader about.
            set(NNTC_LINALG_INSTALL_D3D12 "${NNTC_LINALG_AGILITY}/bin/${NNTC_LINALG_ARCH}/D3D12Core.dll"
                                          "${NNTC_LINALG_AGILITY}/bin/${NNTC_LINALG_ARCH}/d3d12SDKLayers.dll")
            set(NNTC_LINALG_INSTALL_BIN "${NNTC_LINALG_WARP}/bin/${NNTC_LINALG_ARCH}/d3d10warp.dll"
                                        "${NNTC_LINALG_DXC}/bin/${NNTC_LINALG_ARCH}/dxcompiler.dll"
                                        "${NNTC_LINALG_DXC}/bin/${NNTC_LINALG_ARCH}/dxil.dll"
                                        "${CMAKE_CURRENT_SOURCE_DIR}/viewer_d3d12/bin/nntc_view_linalg.hlsl")
            set(NNTC_LINALG_INSTALL_DX "${NNTC_LINALG_DXC}/include/hlsl/dx/linalg.h")
            message(STATUS "nntc_view_d3d12: the linear-algebra path is compiled IN (Agility ${NNTC_LINALG_AGILITY_VERSION}, "
                           "DXC ${NNTC_LINALG_DXC_VERSION}, WARP ${NNTC_LINALG_WARP_VERSION}, ${NNTC_LINALG_ARCH}); whether "
                           "it RUNS is still Developer Mode and the device's own query")
        endif()
    endif()
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
    message(WARNING "Vulkan was not found: the target nntc_view_vk is skipped (everything else still builds). Install "
                    "the Vulkan loader's development package (libvulkan-dev on Debian and Ubuntu, vulkan-loader-devel "
                    "and vulkan-headers on Fedora, vulkan-devel on openSUSE, vulkan-icd-loader and vulkan-headers on "
                    "Arch; the LunarG SDK on Windows) and run cmake again")
elseif(NOT NNTC_SHADERC)
    message(WARNING "No shaderc was found (neither the SDK's shaderc_combined nor a shared libshaderc with its header): "
                    "nntc_view_vk is skipped, since it compiles its shaders at run time. Install the shaderc development "
                    "package (libshaderc-dev on Debian and Ubuntu, libshaderc-devel on Fedora, shaderc-devel on openSUSE, "
                    "shaderc on Arch; on Windows it is part of the LunarG SDK) and run cmake again")
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
                        "The viewer still builds and runs: it decodes with its standard shader, which is the default on "
                        "every GPU; the extension only makes the decode faster on recent NVIDIA cards, so on AMD and "
                        "Intel nothing is missing. For that faster path on an NVIDIA "
                        "card under Linux, build against newer headers: the LunarG SDK (source its setup-env.sh "
                        "before cmake), or a distribution whose vulkan-headers package is 1.4.307 or newer "
                        "(Fedora 42, Arch, Debian 13)")
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

    # The GLSL files travel with the build and are compiled at runtime from BESIDE THE EXECUTABLE and from nowhere
    # else: edit the file in viewer_vk/bin/, build, and press R; an edit made to the copy itself lasts only until the
    # next build overwrites it. The viewer used to look in the working directory first, where a stray view.frag
    # silently replaced the real one, so this copy is now the only file it can read. The copies are their own target,
    # run on every build, as for nntc_view.hlsl above: with a Makefile or Ninja generator a post-build copy ran only when
    # the executable relinked, and a shader-only change stayed behind.
    set(NNTC_VK_SHADERS view.vert view.frag view_coopvec.frag)
    set(NNTC_VK_SHADER_COPY_COMMANDS COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:nntc_view_vk>")
    foreach(NNTC_VK_SHADER ${NNTC_VK_SHADERS})
        list(APPEND NNTC_VK_SHADER_COPY_COMMANDS COMMAND ${CMAKE_COMMAND} -E copy_if_different
             "${CMAKE_CURRENT_SOURCE_DIR}/viewer_vk/bin/${NNTC_VK_SHADER}"
             "$<TARGET_FILE_DIR:nntc_view_vk>/${NNTC_VK_SHADER}")
    endforeach()
    add_custom_target(nntc_view_vk_shaders ALL ${NNTC_VK_SHADER_COPY_COMMANDS} VERBATIM)
    add_dependencies(nntc_view_vk nntc_view_vk_shaders)
endif()

# ---------------------------------------------------------------------------------------------------------------------
# Install: every program that was built, and beside each viewer the shader files it reads from beside itself, so that
# `cmake --install build --prefix DIR` gives a DIR/bin that runs from anywhere; the build tree runs as it is too.
# ---------------------------------------------------------------------------------------------------------------------
foreach(NNTC_PROGRAM nntc_encode nntc_view nntc_view_d3d12 bc_check nntc_view_vk)
    if(TARGET ${NNTC_PROGRAM})
        install(TARGETS ${NNTC_PROGRAM} RUNTIME DESTINATION bin)
    endif()
endforeach()
if(TARGET nntc_view OR TARGET nntc_view_d3d12)
    install(FILES viewer/bin/nntc_view.hlsl DESTINATION bin)   # one file, read by both Direct3D viewers
endif()
if(TARGET nntc_view_d3d12 AND NNTC_LINALG_INSTALL_BIN)
    # The preview runtime travels with the installed Direct3D 12 viewer, in the build tree's own layout: the two
    # Agility files under D3D12\, which is the folder the executable's D3D12SDKPath export names, and d3d10warp.dll
    # BESIDE the executable, which is the one place it is picked up from. Without them an installed copy does not
    # fall back to the plain path - it gets no device at all.
    install(FILES ${NNTC_LINALG_INSTALL_D3D12} DESTINATION bin/D3D12)
    install(FILES ${NNTC_LINALG_INSTALL_BIN} DESTINATION bin)
    install(FILES ${NNTC_LINALG_INSTALL_DX} DESTINATION bin/dx)
endif()
if(TARGET nntc_view_vk)
    list(TRANSFORM NNTC_VK_SHADERS PREPEND "viewer_vk/bin/" OUTPUT_VARIABLE NNTC_VK_SHADER_FILES)
    install(FILES ${NNTC_VK_SHADER_FILES} DESTINATION bin)
endif()
