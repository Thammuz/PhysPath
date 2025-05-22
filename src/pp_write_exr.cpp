// Minimal EXR writer demo  –  Phys-Path
// Requires: find_package(OpenEXR CONFIG REQUIRED)
// Compile flags & include dirs are provided automatically by CMake.

#include <ImfRgbaFile.h>   // OpenEXR main I/O class
#include <ImfRgba.h>
#include <ImathVec.h>
#include <vector>
#include <iostream>

int main()
{
    const int W = 512;
    const int H = 512;

    // OpenEXR's RGBA struct (half-precision by default)
    std::vector<Imf::Rgba> pixels(W * H);

    // Simple blue-to-yellow gradient for testing
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            float u = static_cast<float>(x) / (W - 1);
            float v = static_cast<float>(y) / (H - 1);

            // Imf::Rgba stores components as half, but the ctor accepts float.
            pixels[y * W + x] = Imf::Rgba(
                u,        // R
                v,        // G
                0.2f,     // B
                1.0f      // A
            );
        }
    }

    try
    {
        Imf::RgbaOutputFile file(
            "hello.exr",   // output filename
            W, H,
            Imf::WRITE_RGBA
        );

        // xStride = 1 pixel, yStride = one scanline (W pixels)
        file.setFrameBuffer(pixels.data(), /*xStride*/ 1, /*yStride*/ W);
        file.writePixels(H);

        std::cout << "Wrote hello.exr (" << W << " × " << H << ")\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "OpenEXR write failed: " << e.what() << '\n';
        return 1;
    }

    return 0;
}