#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_VersionVSS. Stores all the properties.
*/
struct VehicleVersionVssData
{
    int32_t m_Major {0};
    int32_t m_Minor {0};
    int32_t m_Patch {0};
    std::string m_Label {std::string()};
};

}
}