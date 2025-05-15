#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_MotionManagement_Brake_Axle_Row2. Stores all the properties.
*/
struct VehicleMotionManagementBrakeAxleRow2Data
{
    int32_t m_TorqueElectricMinimum {0};
    int32_t m_TorqueFrictionDifferenceMaximum {0};
    int32_t m_TorqueDistributionFrictionRightMaximum {0};
    int32_t m_TorqueDistributionFrictionRightMinimum {0};
};

}
}