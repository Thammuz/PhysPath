// gpu_kernels.cu  — OptiX ray-gen kernel with explicit gradient override
// ---------------------------------------------------------------
#include <optix_device.h>        // MUST be first
#include <cuda_runtime.h>
#include "launch_params.h"       // defines LaunchParams and "extern constant params"

// Bring necessary math helpers into global scope
using phys::cam::operator-;
using phys::cam::operator+;
using phys::cam::operator*;
using phys::cam::operator/;
using phys::cam::dot;
using phys::cam::normalize;

// Device constant launch parameters (declared in launch_params.h)
extern "C" __constant__ LaunchParams params;

// ---------------------------------------------------------------------------
// Miss program: invoked when a ray does not hit any geometry
// ---------------------------------------------------------------------------
extern "C" __global__ void __miss__ms()
{
    // no operation: background color is set in raygen
}

// ---------------------------------------------------------------------------
// Sphere intersection helper
// ---------------------------------------------------------------------------
__device__ bool hitSphere(const float3 &center, float radius,
                          const float3 &ro, const float3 &rd,
                          float &tHit, float3 &normal)
{
    float3 oc = ro - center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0f) return false;
    float sqrtD = sqrtf(disc);
    float t0 = -b - sqrtD;
    if (t0 <= 0.0f) return false;
    tHit = t0;
    float3 p = ro + rd * tHit;
    normal = normalize(p - center);
    return true;
}

// ---------------------------------------------------------------------------
// Ray-generation program: primary rays
// ---------------------------------------------------------------------------
extern "C" __global__ void __raygen__rg()
{
    // Compute pixel index
    const uint3 launchIdx = optixGetLaunchIndex();
    const uint32_t pix = launchIdx.y * params.width + launchIdx.x;

    // Generate camera ray
    phys::cam::CameraSample samp;
    samp.pixel = make_float2(
        (launchIdx.x + 0.5f) / params.width,
        (launchIdx.y + 0.5f) / params.height
    );
    samp.lens = make_float2(0.5f, 0.5f);
    phys::cam::Ray ray = phys::cam::generateRay(params.cam, samp);

    // Default background (sky)
    float3 color = make_float3(0.6f, 0.8f, 1.0f);

    // Intersect sphere at origin, radius 1
    float tHit;
    float3 n;
    if (hitSphere(make_float3(0.0f, 0.0f, 0.0f), 1.0f,
                  ray.origin, ray.dir, tHit, n))
    {
        // Shade by normal (visualize normal)
        color = 0.5f * (n + make_float3(1.0f, 1.0f, 1.0f));
    }

    // Write output
    params.pixels[pix] = make_float4(color.x, color.y, color.z, 1.0f);
}
