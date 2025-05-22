cmake_minimum_required(VERSION 3.25)
project(PhysPath LANGUAGES C CXX CUDA)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Always link against the DLL CRT (/MD or /MDd)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# ---------------------------------------------------------------------------
# vcpkg tool-chain is supplied by CMakePresets.json;
#   make sure that file sets:
#       "CMAKE_TOOLCHAIN_FILE":"${sourceDir}/extern/vcpkg/scripts/buildsystems/vcpkg.cmake"
#       "VCPKG_TARGET_TRIPLET":"x64-windows-release" (or x64-windows)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dependencies pulled in via vcpkg
# ---------------------------------------------------------------------------
find_package(embree CONFIG   REQUIRED)          # embree::embree  (4.x)
find_package(OpenEXR CONFIG  REQUIRED)          # OpenEXR::OpenEXR + Imath
find_package(CUDAToolkit REQUIRED)            # CMake 3.17+


# ---------------------------------------------------------------------------
# CPU target 1 – Embree version print
# ---------------------------------------------------------------------------
add_executable(pp_cpu_hello
    src/pp_cpu_hello.cpp)
target_link_libraries(pp_cpu_hello PRIVATE
    embree)
target_compile_definitions(pp_cpu_hello PRIVATE NOMINMAX)

# ---------------------------------------------------------------------------
# CPU target 2 – OpenEXR writer
# ---------------------------------------------------------------------------
add_executable(pp_write_exr
    src/pp_write_exr.cpp)
target_link_libraries(pp_write_exr PRIVATE
    OpenEXR::OpenEXR)
target_compile_definitions(pp_write_exr PRIVATE NOMINMAX)

# ---------------------------------------------------------------------------
# GPU target – OptiX 9  (host exe + external PTX)
# ---------------------------------------------------------------------------
# -- locate OptiX SDK --------------------------------------------------------
if(NOT DEFINED ENV{OPTIX_SDK_HOME})
    message(FATAL_ERROR "OPTIX_SDK_HOME environment variable not set!")
endif()
set(OPTIX_ROOT "$ENV{OPTIX_SDK_HOME}")
if(NOT EXISTS "${OPTIX_ROOT}/include/optix.h")
    message(FATAL_ERROR "Cannot find optix.h under ${OPTIX_ROOT}/include")
endif()

# -- build PTX from gpu_kernels.cu in its OWN step ---------------------------
set(GPU_PTX ${CMAKE_CURRENT_BINARY_DIR}/gpu_kernels.ptx)
add_custom_command(
    OUTPUT  ${GPU_PTX}
    COMMAND ${CMAKE_CUDA_COMPILER}
            --ptx ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_kernels.cu
            -I"${OPTIX_ROOT}/include"
            -o ${GPU_PTX}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_kernels.cu
    COMMENT "Compiling OptiX PTX: gpu_kernels.ptx"
    VERBATIM)

add_custom_target(GpuPTX ALL DEPENDS ${GPU_PTX})

# -- host executable ---------------------------------------------------------
add_executable(pp_gpu_rgb
    src/gpu_main.cpp)              # <-- contains main()

add_dependencies(pp_gpu_rgb GpuPTX)       # PTX built first

target_include_directories(pp_gpu_rgb PRIVATE
    "${OPTIX_ROOT}/include")
#target_link_directories   (pp_gpu_rgb PRIVATE "${OPTIX_ROOT}/lib/64")
target_link_libraries     (pp_gpu_rgb PRIVATE OpenEXR::OpenEXR)

target_compile_definitions(pp_gpu_rgb PRIVATE NOMINMAX)
target_link_libraries(pp_gpu_rgb PRIVATE CUDA::cudart)

# copy PTX next to the executable
add_custom_command(TARGET pp_gpu_rgb POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${GPU_PTX} $<TARGET_FILE_DIR:pp_gpu_rgb>)

# ---------------------------------------------------------------------------
# Export compile_commands.json (nice for clang-tools / VS Code)
# ---------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)