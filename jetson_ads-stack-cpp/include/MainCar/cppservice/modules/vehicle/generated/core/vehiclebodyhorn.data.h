#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Body_Horn. Stores all the properties.
*/
struct VehicleBodyHornData
{
    bool m_IsActive {false};
};

}
}