// gpu_main.cpp  –  minimal OptiX-9 “hello” launch + EXR write
//
// Build assumptions
//   • OPTIX_SDK_HOME   env-var points to OptiX-9 root (with include/, lib/64/)
//   • CUDA Toolkit 12.x present (cuda.h, cuda_runtime.h)
//   • OpenEXR 3 + Imath 3 pulled in via vcpkg   (OpenEXR::OpenEXR target)
//
// CMake must add NOMINMAX and the OptiX dirs, e.g.:
//
//   target_compile_definitions(pp_gpu_rgb PRIVATE NOMINMAX)
//   target_include_directories(pp_gpu_rgb PRIVATE "${OPTIX_ROOT}/include")
//   target_link_directories   (pp_gpu_rgb PRIVATE "${OPTIX_ROOT}/lib/64")
//   target_link_libraries     (pp_gpu_rgb PRIVATE optix OpenEXR::OpenEXR)
//
// The device PTX is loaded at run-time from “gpu_kernels.ptx”.
#define OPTIX_DEFINE_FUNCTIONS     // <--  make the header *define* the symbols
#include <optix_function_table_definition.h>
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include <ImfRgbaFile.h>
#include <ImathVec.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#define checkOptix(call, msg)                                         \
    do {                                                              \
        OptixResult _res = call;                                      \
        if (_res != OPTIX_SUCCESS) {                                  \
            std::cerr << "OptiX error (" << _res << "): " << msg;     \
            std::exit(1);                                             \
        }                                                             \
    } while (0)

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */

static std::string readFile(const char* path)
{
    std::ifstream ifs(path, std::ios::binary);
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

/* launch-param struct must match the one in gpu_kernels.cu */
struct Params
{
    float4* pixels;
    unsigned int width;
    unsigned int height;
};

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */

int main()
{
    constexpr unsigned W = 512, H = 512;
    const size_t pixelCount = W * size_t(H);

    /* ---- OptiX context ------------------------------------------- */
    checkOptix(optixInit(), "optixInit");

    OptixDeviceContext context = nullptr;
    checkOptix(optixDeviceContextCreate(0, nullptr, &context),
        "device context create");

    /* ---- load PTX and create module ------------------------------ */
    const std::string ptx = readFile("gpu_kernels.ptx");

    OptixPipelineCompileOptions compileOpts = {};
    compileOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    compileOpts.usesMotionBlur = 0;
    compileOpts.numPayloadValues = 2;
    compileOpts.numAttributeValues = 2;
    compileOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    compileOpts.pipelineLaunchParamsVariableName = "params";

    OptixModuleCompileOptions mOpts = {};          // defaults fine
    OptixModule module = nullptr;

    checkOptix(optixModuleCreate(
        context,              // OptixDeviceContext
        &mOpts,               // OptixModuleCompileOptions
        &compileOpts,         // OptixPipelineCompileOptions
        ptx.c_str(),          // pointer to PTX string
        ptx.size(),           // PTX size in bytes
        nullptr, nullptr,     // (optional) log buffer + size
        &module), "module create");

    /* ---- program group (ray-gen only) ---------------------------- */
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__rg";

    OptixProgramGroup pg = nullptr;
    checkOptix(optixProgramGroupCreate(
        context, &pgDesc, 1,
        nullptr, nullptr, nullptr, &pg),
        "program group create");

    /* ---- pipeline ------------------------------------------------ */
    OptixProgramGroup pgs[] = { pg };

    OptixPipelineLinkOptions linkOpts = {};
    linkOpts.maxTraceDepth = 1;
    //linkOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipeline pipeline = nullptr;
    checkOptix(optixPipelineCreate(
        context,
        &compileOpts,
        &linkOpts,
        pgs, 1,
        nullptr, nullptr,
        &pipeline),
        "pipeline create");

    /* ---- shader binding table (single RG record) ----------------- */
    CUdeviceptr  d_raygenRecord = 0;
    const size_t sbtSize = OPTIX_SBT_RECORD_HEADER_SIZE;
    cudaMalloc(reinterpret_cast<void**>(&d_raygenRecord), sbtSize);
    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = d_raygenRecord;

    /* ---- launch params & pixel buffer ---------------------------- */
    float4* d_pixels = nullptr;
    cudaMalloc(&d_pixels, pixelCount * sizeof(float4));

    Params h_params{ d_pixels, W, H };
    Params* d_params = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params));
    cudaMemcpy(d_params, &h_params, sizeof(Params), cudaMemcpyHostToDevice);

    /* ---- launch -------------------------------------------------- */
    checkOptix(optixLaunch(
        pipeline, 0,                   // stream = 0
        reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(Params),
        &sbt,
        W, H, 1),
        "optixLaunch");
    cudaDeviceSynchronize();

    /* ---- read back & write EXR ----------------------------------- */
    std::vector<Imf::Rgba> hostPixels(pixelCount);
    cudaMemcpy(hostPixels.data(), d_pixels,
        pixelCount * sizeof(float4), cudaMemcpyDeviceToHost);

    try {
        Imf::RgbaOutputFile exr("gpu_hello.exr", W, H, Imf::WRITE_RGBA);
        exr.setFrameBuffer(hostPixels.data(), 1, W);
        exr.writePixels(H);
        std::cout << "Wrote gpu_hello.exr (" << W << " × " << H << ")\n";
    }
    catch (const std::exception& e) {
        std::cerr << "EXR write failed: " << e.what() << '\n';
    }

    /* ---- cleanup ------------------------------------------------- */
    cudaFree(d_pixels);
    cudaFree(reinterpret_cast<void*>(d_raygenRecord));
    cudaFree(d_params);
    optixPipelineDestroy(pipeline);
    optixProgramGroupDestroy(pg);
    optixModuleDestroy(module);
    optixDeviceContextDestroy(context);
    return 0;
}
