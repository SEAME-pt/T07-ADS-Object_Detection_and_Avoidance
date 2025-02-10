#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Body_Lights_Beam_High. Stores all the properties.
*/
struct VehicleBodyLightsBeamHighData
{
    bool m_IsOn {false};
    bool m_IsDefect {false};
};

}
}