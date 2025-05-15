#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Diagnostics. Stores all the properties.
*/
struct VehicleDiagnosticsData
{
    int32_t m_DTCCount {0};
    std::list<std::string> m_DTCList {std::list<std::string>()};
};

}
}