// src/usd_loader.cpp — Minimal USD scene loader to import meshes for OptiX
// ---------------------------------------------------------------
// Requires linking against Pixar USD (usd, usdGeom, tf, sdf, etc.)

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <vector>
#include <string>
#include <iostream>

// Simple structure to hold mesh buffers
struct MeshData {
    std::vector<float> positions; // x,y,z triples
    std::vector<float> normals;   // x,y,z triples
    std::vector<uint32_t> indices;
    std::string name;
};

// Load all meshes from a USD file into MeshData objects
std::vector<MeshData> LoadUsdMeshes(const std::string& usdFilePath) {
    // Open the USD stage
    pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(usdFilePath);
    if (!stage) {
        throw std::runtime_error("Failed to open USD stage: " + usdFilePath);
    }

    // Cache for local-to-world transforms if needed
    pxr::UsdGeomXformCache xformCache(stage);

    std::vector<MeshData> meshes;

    // Iterate over all UsdGeomMesh prims
    for (const auto& prim : stage->Traverse()) {
        if (!prim.IsA<pxr::UsdGeomMesh>()) continue;
        pxr::UsdGeomMesh mesh(prim);

        // Extract points
        pxr::VtArray<pxr::GfVec3f> pts;
        mesh.GetPointsAttr().Get(&pts);

        // Extract normals (optional)
        pxr::VtArray<pxr::GfVec3f> nmls;
        mesh.GetNormalsAttr().Get(&nmls);

        // Extract face vertex counts & indices
        pxr::VtArray<int> faceVertexCounts;
        pxr::VtArray<int> faceVertexIndices;
        mesh.GetFaceVertexCountsAttr().Get(&faceVertexCounts);
        mesh.GetFaceVertexIndicesAttr().Get(&faceVertexIndices);

        // Convert to flat arrays
        MeshData data;
        data.name = mesh.GetPrim().GetName().GetString();
        data.positions.reserve(pts.size() * 3);
        for (auto& p : pts) {
            data.positions.push_back(p[0]);
            data.positions.push_back(p[1]);
            data.positions.push_back(p[2]);
        }
        data.normals.reserve(nmls.size() * 3);
        for (auto& n : nmls) {
            data.normals.push_back(n[0]);
            data.normals.push_back(n[1]);
            data.normals.push_back(n[2]);
        }
        data.indices.reserve(faceVertexIndices.size());
        for (auto idx : faceVertexIndices) {
            data.indices.push_back(static_cast<uint32_t>(idx));
        }

        meshes.push_back(std::move(data));
    }

    return meshes;
}

// Example usage
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: usd_loader <scene.usd>" << std::endl;
        return 1;
    }
    std::string usdPath = argv[1];

    try {
        auto meshes = LoadUsdMeshes(usdPath);
        std::cout << "Loaded " << meshes.size() << " meshes from " << usdPath << std::endl;
        for (auto& m : meshes) {
            std::cout << "Mesh '" << m.name << "': "
                << m.positions.size() / 3 << " verts, "
                << m.indices.size() / 3 << " triangles" << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
