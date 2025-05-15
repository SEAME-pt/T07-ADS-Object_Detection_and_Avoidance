#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Body_Lights_Brake. Stores all the properties.
*/
struct VehicleBodyLightsBrakeData
{
    Vehicle_Body_Lights_Brake_IsActive_ValueEnum m_IsActive {Vehicle_Body_Lights_Brake_IsActive_ValueEnum::INACTIVE};
    bool m_IsDefect {false};
};

}
}