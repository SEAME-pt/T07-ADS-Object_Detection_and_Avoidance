#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Connectivity. Stores all the properties.
*/
struct VehicleConnectivityData
{
    bool m_IsConnectivityAvailable {false};
};

}
}