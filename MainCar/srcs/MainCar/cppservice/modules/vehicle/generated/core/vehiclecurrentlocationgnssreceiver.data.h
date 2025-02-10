#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_CurrentLocation_GNSSReceiver. Stores all the properties.
*/
struct VehicleCurrentLocationGnssReceiverData
{
    Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum m_FixType {Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum::NONE};
};

}
}