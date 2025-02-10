#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_CurrentLocation_GNSSReceiver_MountingPosition. Stores all the properties.
*/
struct VehicleCurrentLocationGnssReceiverMountingPositionData
{
    int32_t m_X {0};
    int32_t m_Y {0};
    int32_t m_Z {0};
};

}
}