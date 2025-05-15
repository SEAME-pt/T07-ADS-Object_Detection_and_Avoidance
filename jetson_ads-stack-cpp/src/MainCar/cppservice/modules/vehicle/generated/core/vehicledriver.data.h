#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Driver. Stores all the properties.
*/
struct VehicleDriverData
{
    float m_DistractionLevel {0.0f};
    bool m_IsEyesOnRoad {false};
    bool m_IsHandsOnWheel {false};
    float m_AttentiveProbability {0.0f};
    float m_FatigueLevel {0.0f};
    int32_t m_HeartRate {0};
};

}
}