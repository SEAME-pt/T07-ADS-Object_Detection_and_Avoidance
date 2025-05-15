#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_MotionManagement_Brake. Stores all the properties.
*/
struct VehicleMotionManagementBrakeData
{
    int32_t m_VehicleForceMaximum {0};
    int32_t m_VehicleForceElectricMinimumArbitrated {0};
    int32_t m_VehicleForceElectric {0};
    int32_t m_VehicleForceDistributionFrontMaximum {0};
    int32_t m_VehicleForceDistributionFrontMinimum {0};
};

}
}