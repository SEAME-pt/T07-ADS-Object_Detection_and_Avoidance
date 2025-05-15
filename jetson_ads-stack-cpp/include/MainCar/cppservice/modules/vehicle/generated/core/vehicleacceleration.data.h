#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Acceleration. Stores all the properties.
*/
struct VehicleAccelerationData
{
    float m_Longitudinal {0.0f};
    float m_Lateral {0.0f};
    float m_Vertical {0.0f};
};

}
}