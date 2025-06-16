cmake_minimum_required(VERSION 3.18)
project(PhysPath LANGUAGES CXX CUDA)

# -------------------------------------------------------------
# 0. Compiler & global flags
# -------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------------------------------------------
# 1. Core dependencies (resolved via the vcpkg tool?chain file)
# -------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)            # modern CUDA
find_package(OpenEXR      REQUIRED CONFIG)   # OpenEXR::OpenEXR

# OptiX headers (no linking)
find_path(OPTIX_INCLUDE_DIR optix.h HINTS "$ENV{OPTIX_SDK_HOME}/include")

include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPTIX_INCLUDE_DIR}
    src                    # local headers (camera/, etc.)
)

# -------------------------------------------------------------
# 2. Optional Pixar USD (via vcpkg) — don’t fail if absent
# -------------------------------------------------------------
find_package(usd QUIET CONFIG COMPONENTS usd usdGeom tf sdf)
if(usd_FOUND)
    message(STATUS "USD found – enabling usd_loader target and USD support in renderer")
else()
    message(STATUS "USD NOT found – renderer will build without USD; usd_loader skipped")
endif()

# -------------------------------------------------------------
# 3. Main renderer executable
# -------------------------------------------------------------
add_executable(pp_gpu_rgb src/gpu_main.cpp)

target_compile_definitions(pp_gpu_rgb PRIVATE NOMINMAX)

target_link_libraries(pp_gpu_rgb PRIVATE
    CUDA::cuda_driver
    OpenEXR::OpenEXR)

if(usd_FOUND)
    target_link_libraries(pp_gpu_rgb PRIVATE
        usd::usd usd::usdGeom usd::tf usd::sdf)
endif()

# -------------------------------------------------------------
# 4. USD mesh?dump utility (only if USD present)
# -------------------------------------------------------------
if(usd_FOUND)
    add_executable(usd_loader src/usd_loader.cpp)
    target_link_libraries(usd_loader PRIVATE
        usd::usd usd::usdGeom usd::tf usd::sdf)
    install(TARGETS usd_loader RUNTIME DESTINATION bin)
endif()

# -------------------------------------------------------------
# 5. Install renderer
# -------------------------------------------------------------
install(TARGETS pp_gpu_rgb RUNTIME DESTINATION bin)
