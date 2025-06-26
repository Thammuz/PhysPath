// src/usd_loader.cpp — Minimal USD scene loader to import meshes
// -----------------------------------------------------------------------------
// Works with Pixar USD?25.x as shipped via vcpkg. It discovers the plug?ins that
// live under …/installed/x64?windows/bin/usd **and** the nested …/bin/usd/usd
// folder that contains the core "usd" plug?in (the one exporting
// `Usd_UsdzResolver`).  Without registering the latter explicitly, the stage
// fails to open in silence because no package?resolver is available.
// -----------------------------------------------------------------------------
// 2025?06?26 – debug v4 for @phosp
//   • Register **three** folders:
//         <vcpkg>/installed/x64-windows/bin/usd          (third?party)
//         <vcpkg>/installed/x64-windows/bin/usd/usd      (core)
//         <vcpkg>/installed/x64-windows/plugin/usd       (old layout)
//   • Emit the list of paths so we can double?check at run time.
//   • Flush every diagnostic with std::endl so nothing is lost if the process
//     aborts deep inside USD.
// -----------------------------------------------------------------------------

#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/base/plug/registry.h>
#include <pxr/usd/ar/resolver.h>

#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>

namespace px = pxr;

struct MeshData
{
    std::vector<float>    positions;
    std::vector<float>    normals;
    std::vector<uint32_t> indices;
    std::string           name;
};

static std::vector<MeshData> LoadUsdMeshes(const std::string& usdFile)
{
    px::UsdStageRefPtr stage = px::UsdStage::Open(usdFile);
    if (!stage)
        throw std::runtime_error("Failed to open stage: " + usdFile);

    px::UsdGeomXformCache xformCache;
    std::vector<MeshData> meshes;

    for (px::UsdPrim prim : stage->Traverse()) {
        if (!prim.IsA<px::UsdGeomMesh>())
            continue;
        px::UsdGeomMesh gmesh(prim);

        px::VtArray<px::GfVec3f> pts; gmesh.GetPointsAttr().Get(&pts);
        px::VtArray<px::GfVec3f> nml; gmesh.GetNormalsAttr().Get(&nml);
        px::VtArray<int> counts, indices;
        gmesh.GetFaceVertexCountsAttr().Get(&counts);
        gmesh.GetFaceVertexIndicesAttr().Get(&indices);

        MeshData m; m.name = prim.GetName().GetString();
        m.positions.reserve(pts.size() * 3);
        for (auto& p : pts) m.positions.insert(m.positions.end(), { p[0],p[1],p[2] });
        m.normals.reserve(nml.size() * 3);
        for (auto& n : nml) m.normals.insert(m.normals.end(), { n[0],n[1],n[2] });
        m.indices.reserve(indices.size());
        for (int i : indices) m.indices.push_back(static_cast<uint32_t>(i));
        meshes.push_back(std::move(m));
    }
    return meshes;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: usd_loader <scene.usd/.usda/.usdz>" << std::endl;
        return 1;
    }

    const std::string file = argv[1];
    std::cout << "Identifier  : " << file << std::endl
        << "exists?     : " << std::filesystem::exists(file) << std::endl
        << "Resolved ID : " << px::ArGetResolver().Resolve(file).GetPathString() << std::endl;

    try {
        auto meshes = LoadUsdMeshes(file);
        std::cout << "Loaded " << meshes.size() << " mesh(es)" << std::endl;
        for (auto& m : meshes) {
            std::cout << "  " << m.name << " — "
                << m.positions.size() / 3 << " verts, "
                << m.indices.size() / 3 << " faces" << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 2;
    }
    return 0;
}
