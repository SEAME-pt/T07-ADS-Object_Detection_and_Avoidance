#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Body_Lights. Stores all the properties.
*/
struct VehicleBodyLightsData
{
    Vehicle_Body_Lights_LightSwitch_ValueEnum m_LightSwitch {Vehicle_Body_Lights_LightSwitch_ValueEnum::OFF};
    bool m_IsHighBeamSwitchOn {false};
};

}
}