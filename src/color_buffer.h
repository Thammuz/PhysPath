#pragma once
// Simple helpers for packing / unpacking float4 pixels.
// CUDA already defines float2/float3/float4 in <cuda_runtime.h>, so we
// only need a convenience constructor here.

#include <cuda_runtime.h>   // brings in float4, make_float4

// Utility: convert four floats to a packed 32-bit RGBA8 (optional)
__host__ __device__ inline uint32_t packRGBA8(float r, float g, float b, float a = 1.f)
{
    auto to8 = [](float v) { return static_cast<uint32_t>(fminf(fmaxf(v, 0.f), 1.f) * 255.f + 0.5f); };
    return  (to8(a) << 24) | (to8(b) << 16) | (to8(g) << 8) | to8(r);
}