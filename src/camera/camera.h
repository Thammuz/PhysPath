// camera.h  — PhysPath unified camera interface (no duplicate make_float3)
#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace phys {
    namespace cam {

        // ---------------- vector helpers ----------------
        // Use CUDA's built?in make_float{2,3}. We only add scalar ops.

        __host__ __device__ inline float3 operator+(const float3& a, const float3& b) { return float3{ a.x + b.x,a.y + b.y,a.z + b.z }; }
        __host__ __device__ inline float3 operator-(const float3& a, const float3& b) { return float3{ a.x - b.x,a.y - b.y,a.z - b.z }; }
        __host__ __device__ inline float3 operator*(const float3& v, float s) { return float3{ v.x * s,v.y * s,v.z * s }; }
        __host__ __device__ inline float3 operator*(float s, const float3& v) { return v * s; }
        __host__ __device__ inline float3 operator/(const float3& v, float s) { float inv = 1.f / s; return v * inv; }
        __host__ __device__ inline float3& operator*=(float3& v, float s) { v.x *= s; v.y *= s; v.z *= s; return v; }
        __host__ __device__ inline float3& operator/=(float3& v, float s) { float inv = 1.f / s; v.x *= inv; v.y *= inv; v.z *= inv; return v; }

        __host__ __device__ inline float dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
        __host__ __device__ inline float3 cross(const float3& a, const float3& b) { return float3{ a.y * b.z - a.z * b.y,a.z * b.x - a.x * b.z,a.x * b.y - a.y * b.x }; }
        __host__ __device__ inline float length(const float3& v) { return sqrtf(dot(v, v)); }
        __host__ __device__ inline float3 normalize(const float3& v) { return v / length(v); }

        struct Ray { float3 origin; float3 dir; };
        struct CameraSample { float2 pixel; float2 lens; };

        struct PinholeCamera {
            float3 pos; float3 forward; float3 right; float3 up;
        };

        __host__ inline PinholeCamera makePinhole(float3 eye, float3 look, float3 upWorld, float vfovDeg, float aspect) {
            PinholeCamera cam{};
            cam.pos = eye;
            cam.forward = normalize(float3{ look.x - eye.x, look.y - eye.y, look.z - eye.z });
            // Build orthonormal basis; guard against upWorld parallel to forward
            cam.right = cross(cam.forward, upWorld);
            if (length(cam.right) < 1e-6f) {
                // choose an arbitrary axis if forward is nearly parallel to upWorld
                cam.right = cross(cam.forward, float3{ 1.0f, 0.0f, 0.0f });
                if (length(cam.right) < 1e-6f) {
                    cam.right = cross(cam.forward, float3{ 0.0f, 1.0f, 0.0f });
                }
            }
            cam.right = normalize(cam.right);
            cam.up = cross(cam.right, cam.forward);

            float tanHalf = tanf(0.5f * vfovDeg * 3.14159265f / 180.0f);
            cam.up *= tanHalf;
            cam.right *= tanHalf * aspect;
            return cam;
        }

        __host__ __device__ inline Ray generateRay(const PinholeCamera& cam, const CameraSample& s) {
            float2 ndc = float2{ 2.f * s.pixel.x - 1.f,2.f * s.pixel.y - 1.f };
            float3 dir = normalize(cam.forward + ndc.x * cam.right + ndc.y * cam.up);
            return Ray{ cam.pos,dir };
        }

    } // namespace cam
} // namespace phys
