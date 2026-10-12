#include "prelude.hpp"
#include <m-detector/DynObjFilter.h>
#include <cstdint>
// Compile-only ELF metadata. No backend/filter/cluster object is constructed,
// no executable is linked, and none of the following types' methods is called.
extern "C" __attribute__((section(".mdetector_layout"),used))
const std::uint64_t mdetector_layout[] = {
    0x4d44455441424931ULL, 1,
    sizeof(point_soph), alignof(point_soph),
    sizeof(Point_Cloud), alignof(Point_Cloud),
    sizeof(DepthMap), alignof(DepthMap),
    sizeof(DynObjFilter), sizeof(DynObjCluster),
    sizeof(PointType), alignof(PointType),
    sizeof(point_soph*), sizeof(std::vector<point_soph*>),
    sizeof(float), sizeof(int), sizeof(double),
};
