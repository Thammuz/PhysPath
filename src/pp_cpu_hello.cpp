#include <embree3/rtcore.h>   // header supplied by the vcpkg embree port
#include <iostream>

int main()
{
    /* Create a default device (nullptr = use env variables / defaults) */
    RTCDevice device = rtcNewDevice(nullptr);
    if (!device) {
        std::cerr << "Embree device creation failed\n";
        return 1;
    }

    /* Embree 4: query properties via rtcGetDeviceProperty */
    int version = rtcGetDeviceProperty(device, RTC_DEVICE_PROPERTY_VERSION);
    std::cout << "Embree version: " << version << '\n';

    rtcReleaseDevice(device);
    return 0;
}