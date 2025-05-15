#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_AngularVelocity. Stores all the properties.
*/
struct VehicleAngularVelocityData
{
    float m_Roll {0.0f};
    float m_Pitch {0.0f};
    float m_Yaw {0.0f};
};

}
}