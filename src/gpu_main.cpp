// gpu_main.cpp – Host-side application for OptiX path tracer proof-of-concept
// ---------------------------------------------------------------
#include <cuda_runtime.h>
#include <optix.h>
#define OPTIX_DEFINE_FUNCTIONS
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <OpenEXR/ImfRgbaFile.h>
#include <Imath/half.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "camera/camera.h"      // phys::cam::PinholeCamera
#include "gpu/launch_params.h"      // LaunchParams + extern __constant__ params

#define CHK(call)                                                            \
  do { OptixResult _r = (call);                                             \
       if (_r != OPTIX_SUCCESS) {                                           \
           std::cerr << "OptiX error " << _r                               \
                     << " at " << __FILE__ << ':' << __LINE__             \
                     << std::endl;                                          \
           std::exit(1); } } while(0)

// Utility: read entire file into a string
static std::string readFile(const char* path) {
    std::ifstream fs(path, std::ios::binary);
    std::ostringstream ss; ss << fs.rdbuf();
    return ss.str();
}

int main() {
    constexpr unsigned W = 512, H = 512;
    const size_t pixelCount = size_t(W) * size_t(H);

    // 0. CUDA memcpy test (validate buffer transfers)
    {
        float4* d_test = nullptr;
        cudaMalloc(&d_test, pixelCount * sizeof(float4));
        std::vector<float4> hostTest(pixelCount, make_float4(0.25f, 0.5f, 0.75f, 1.0f));
        cudaMemcpy(d_test, hostTest.data(), pixelCount * sizeof(float4), cudaMemcpyHostToDevice);
        std::vector<float4> testBuf(pixelCount);
        cudaMemcpy(testBuf.data(), d_test, pixelCount * sizeof(float4), cudaMemcpyDeviceToHost);
        std::cout << "Memcpy test – first pixel = "
            << testBuf[0].x << ", "
            << testBuf[0].y << ", "
            << testBuf[0].z << ", "
            << testBuf[0].w << std::endl;
        cudaFree(d_test);
    }

    // 1. Initialize CUDA runtime and OptiX
    cudaFree(0);
    CHK(optixInit());
    OptixDeviceContextOptions ctxOpts = {};
    ctxOpts.logCallbackFunction = [](unsigned lvl, const char*, const char* msg, void*) {
        std::cerr << "[OptiX] " << lvl << ": " << msg << std::endl;
        };
    ctxOpts.logCallbackLevel = 4;
    OptixDeviceContext context = nullptr;
    CHK(optixDeviceContextCreate(0, &ctxOpts, &context));
    std::cout << "OptiX context created" << std::endl;

    // 2. Camera and device buffer
    phys::cam::PinholeCamera cam = phys::cam::makePinhole(
        make_float3(0, 0, 5), make_float3(0, 0, 0), make_float3(0, 1, 0), 45.0f, float(W) / H);
    float4* d_pixels = nullptr;
    cudaMalloc(&d_pixels, pixelCount * sizeof(float4));
    LaunchParams hParams{ cam, d_pixels, W, H };
    LaunchParams* d_params = nullptr;
    cudaMalloc(&d_params, sizeof(LaunchParams));
    cudaMemcpy(d_params, &hParams, sizeof(hParams), cudaMemcpyHostToDevice);

    // 3. Compile PTX module
    std::string ptx = readFile("gpu_kernels.ptx");
    char modLog[4096]; size_t modLogSize = sizeof(modLog);
    OptixModuleCompileOptions mcOpts = {};
    OptixPipelineCompileOptions pcOpts = {};
    pcOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pcOpts.pipelineLaunchParamsVariableName = "params";
    OptixModule module = nullptr;
    CHK(optixModuleCreate(context, &mcOpts, &pcOpts,
        ptx.c_str(), ptx.size(), modLog, &modLogSize, &module));
    if (modLogSize) std::cerr << "Module log:\n" << modLog << std::endl;
    std::cout << "PTX module OK" << std::endl;
    // --- Debug print: before creating raygen program group ---
    std::cout << "Creating Ray-Gen Program Group..." << std::endl;

    // 4a. Ray-gen program group
    OptixProgramGroupDesc rgDesc = {};
    rgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgDesc.raygen.module = module;
    rgDesc.raygen.entryFunctionName = "__raygen__rg";

    OptixProgramGroupOptions pgOpts = {};
    OptixProgramGroup rgPg = nullptr;
    char rgLog[2048]; size_t rgLogSize = sizeof(rgLog);
    OptixResult rgRes = optixProgramGroupCreate(
        context,
        &rgDesc,
        1,
        &pgOpts,
        rgLog, &rgLogSize,
        &rgPg
    );
    std::cerr << "Raygen PG result = " << rgRes << std::endl;
    if (rgLogSize) std::cerr << "Raygen PG log:" << rgLog << std::endl;
        if (rgRes != OPTIX_SUCCESS) {
            std::cerr << "Failed to create Ray-Gen Program Group" << std::endl;
            return 1;
        }
    std::cout << "Raygen PG OK" << std::endl;
    // --- Debug print: before creating miss program group ---
    std::cout << "Creating Miss Program Group..." << std::endl;

    // 4b. Miss program group
    std::cout << "Creating Miss Program Group..." << std::endl;
    OptixProgramGroupDesc msDesc = {};
    msDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msDesc.miss.module = module;
    msDesc.miss.entryFunctionName = "__miss__ms";

    OptixProgramGroupOptions msOpts = {};
    OptixProgramGroup msPg = nullptr;
    char msLog[2048]; size_t msLogSize = sizeof(msLog);
    OptixResult msRes = optixProgramGroupCreate(
        context,
        &msDesc,
        1,
        &msOpts,
        msLog, &msLogSize,
        &msPg
    );
    std::cerr << "Miss PG result = " << msRes << std::endl;
    if (msLogSize) std::cerr << "Miss PG log:" << msLog << std::endl;
        if (msRes != OPTIX_SUCCESS) {
            std::cerr << "Failed to create Miss Program Group" << std::endl;
            return 1;
        }
    std::cout << "Miss PG OK" << std::endl;
    // --- Debug print: before pipeline creation ---
    std::cout << "Building Pipeline..." << std::endl;

    // 5. Build pipeline and set stack size
    OptixPipelineLinkOptions linkOpts = {};
    linkOpts.maxTraceDepth = 1;
    OptixProgramGroup pgs[] = { rgPg,msPg };
    OptixPipeline pipeline = nullptr;
    CHK(optixPipelineCreate(context, &pcOpts, &linkOpts, pgs, 2, nullptr, nullptr, &pipeline));
    CHK(optixPipelineSetStackSize(pipeline, 0u, 0u, 0u, 1u));
    std::cout << "Pipeline & stack OK" << std::endl;
    // --- Debug print: before launch ---
    std::cout << "Launching OptiX..." << std::endl;

    // 6. Build SBT
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; } rgRec, msRec;
    optixSbtRecordPackHeader(rgPg, &rgRec);
    optixSbtRecordPackHeader(msPg, &msRec);
    CUdeviceptr dRg, dMs;
    cudaMalloc((void**)&dRg, sizeof(rgRec));
    cudaMalloc((void**)&dMs, sizeof(msRec));
    cudaMemcpy((void*)dRg, &rgRec, sizeof(rgRec), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)dMs, &msRec, sizeof(msRec), cudaMemcpyHostToDevice);
    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = dRg;
    sbt.missRecordBase = dMs;
    sbt.missRecordCount = 1;
    sbt.missRecordStrideInBytes = sizeof(msRec);

    // 7. Launch OptiX
    CHK(optixLaunch(pipeline, 0,
        reinterpret_cast<CUdeviceptr>(d_params), sizeof(LaunchParams),
        &sbt, W, H, 1));
    cudaDeviceSynchronize();
    std::cout << "optixLaunch completed" << std::endl;
    // --- Debug print: after launch ---
    std::cout << "OptiX launch done, proceeding to debug copy..." << std::endl;

    // 8. Debug first pixel
    std::vector<float4> dbg(pixelCount);
    cudaMemcpy(dbg.data(), d_pixels, pixelCount * sizeof(float4), cudaMemcpyDeviceToHost);
    std::cout << "OptiX pixel[0] = "
        << dbg[0].x << ", " << dbg[0].y << ", " << dbg[0].z << ", " << dbg[0].w << std::endl;

    // 9. Write EXR output
    std::cout << "Writing EXR to gpu_hello.exr..." << std::endl;
    Imf::RgbaOutputFile out("gpu_hello.exr", W, H, Imf::WRITE_RGBA);
    std::vector<Imf::Rgba> buf(pixelCount);
    for (size_t i = 0; i < pixelCount; ++i) { auto& p = dbg[i]; buf[i] = Imf::Rgba(Imath::half(p.x), Imath::half(p.y), Imath::half(p.z), Imath::half(p.w)); }
    out.setFrameBuffer(buf.data(), 1, W);
    out.writePixels(H);
    std::cout << "Finished writing gpu_hello.exr" << std::endl;

    // Cleanup
    cudaFree(d_pixels);
    cudaFree(d_params);
    cudaFree((void*)dRg);
    cudaFree((void*)dMs);
    optixPipelineDestroy(pipeline);
    optixProgramGroupDestroy(rgPg);
    optixProgramGroupDestroy(msPg);
    optixModuleDestroy(module);
    optixDeviceContextDestroy(context);
    return 0;
}
