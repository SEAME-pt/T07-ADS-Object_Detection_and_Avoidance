#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Body. Stores all the properties.
*/
struct VehicleBodyData
{
    int32_t m_PowerOptimizeLevel {0};
    bool m_IsAutoPowerOptimize {false};
};

}
}