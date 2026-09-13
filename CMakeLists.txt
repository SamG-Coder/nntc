cmake_minimum_required(VERSION 3.20)
project(nntc LANGUAGES CXX CUDA)

# The encoder is CUDA only: the host does I/O, the layout, the small dense solves and the report, and every per-texel pass
# is a kernel. It builds on Windows (MSVC + nvcc) and on Linux (gcc / clang + nvcc). The viewer is Direct3D 11 and is
# therefore built on Windows only.
#
# Configure with the Visual Studio generator and a CUDA toolset new enough for the target architectures, for example:
#   cmake -B build -S . -G "Visual Studio 17 2022" -T cuda=13.1
#   cmake --build build --config Release

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    src/bc_pack.h
    src/sample.cuh
    src/device.cuh
    # The material JSON's reader, shared with the viewer and bc_check, which read the ASSET's JSON with the same two
    # headers. It moved out of viewer/ when the encoder gained a reason to parse JSON of its own.
    src/nntc_json.h
    src/json.h
    # The vendored single-header libraries, listed so that an IDE shows what the build actually compiles. All three or
    # none: the resizer became load-bearing when the source chain gained its filters, and a list that named two of the
    # three would read as a statement about the third.
    src/stb_image.h
    src/stb_image_write.h
    src/stb_image_resize2.h)

target_include_directories(nntc_encode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
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

# ---------------------------------------------------------------------------------------------------------------------
# The viewer: Direct3D 11, Windows only, plain C++17 with no dependency beyond the C runtime, Win32, d3d11 and d3dcompiler.
# ---------------------------------------------------------------------------------------------------------------------
if(WIN32)
    # src/bc_pack.h is the one BC4 / BC5 encoder in the tree: the encoder packs level 0 with it and the viewer packs
    # an uncompressed level 0 with it at load, so the two produce the same bytes by construction.
    add_executable(nntc_view viewer/main.cpp src/bc_pack.h src/nntc_json.h src/json.h)
    target_include_directories(nntc_view PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(nntc_view PRIVATE d3d11 d3dcompiler)

    # bc_check packs a .dds the way the viewer packs level 0 and decodes it with iOrange's bcdec (MIT / Unlicense).
    add_executable(bc_check viewer/bc_check.cpp src/bc_pack.h src/nntc_json.h src/json.h viewer/bcdec.h)
    target_include_directories(bc_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

    # The same warning level as the encoder. src/json.h and viewer/bcdec.h are vendored and are included with the
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
