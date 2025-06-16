#include "../camera/camera.h"

struct LaunchParams {
    phys::cam::PinholeCamera cam;
    float4* pixels;                 // RGBA32F buffer on device
    unsigned width, height;
};  