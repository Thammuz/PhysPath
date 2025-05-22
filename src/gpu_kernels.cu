// 1.  MUST be first – brings in all OptiX device intrinsics
#include <optix_device.h>

// 2.  CUDA helpers (float4, make_float4, etc.)
#include <cuda_runtime.h>

// 3.  Your own headers (optional)
#include "color_buffer.h"

// ------------------------------------------------------------------
// Launch-param struct MUST match the one you pass from the host
// ------------------------------------------------------------------
extern "C" {
struct Params
{
    float4*      pixels;
    unsigned int width;
    unsigned int height;
};
__constant__ Params params;
}

// ------------------------------------------------------------------
// Ray-gen program: write orange pixel
// ------------------------------------------------------------------
extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();        // now resolves
    const uint32_t i = idx.y * params.width + idx.x;
    params.pixels[i] = make_float4(1.0f, 0.3f, 0.05f, 1.0f);
}

extern "C" __global__ void __miss__ms()        {}
extern "C" __global__ void __closesthit__ch()  {}