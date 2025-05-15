#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Body_Lights_DirectionIndicator_Left. Stores all the properties.
*/
struct VehicleBodyLightsDirectionIndicatorLeftData
{
    bool m_IsSignaling {false};
    bool m_IsDefect {false};
};

}
}