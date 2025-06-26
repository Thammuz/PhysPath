#include "RegisterVcpkgUsdPlugins.h"

#include <pxr/base/plug/registry.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace px = pxr;

// -------------- original helper ------------------------------------------------
void RegisterVcpkgUsdPlugins()
{
    const char* root = std::getenv("VCPKG_ROOT");
    if (!root) {
        std::cerr << "VCPKG_ROOT not set — skipping automatic plug-in registration\n";
        return;
    }

    std::string base = std::string(root) + "/installed/x64-windows";
    std::string binUsd = base + "/bin/usd";
    std::string coreUsd = binUsd + "/usd";
    std::string plugUsd = base + "/plugin/usd";

    std::vector<std::string> paths = { binUsd, coreUsd, plugUsd };
    px::PlugRegistry::GetInstance().RegisterPlugins(paths);

    std::cout << "Plug-in search paths registered:\n";
    for (auto& p : paths) std::cout << "  • " << p << '\n';
}

// -------------- *this* object triggers the call before main() ------------------
namespace {
    struct _Bootstrap {
        _Bootstrap() { 
            std::cerr << "[BOOT] static initializer ran\n" << std::endl;
            RegisterVcpkgUsdPlugins(); }
    };
    static _Bootstrap _run;            // global – ctor runs at load time
}
