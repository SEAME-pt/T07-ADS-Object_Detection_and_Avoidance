#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_CurrentLocation. Stores all the properties.
*/
struct VehicleCurrentLocationData
{
    std::string m_Timestamp {std::string()};
    double m_Latitude {0.0};
    double m_Longitude {0.0};
    double m_Heading {0.0};
    double m_HorizontalAccuracy {0.0};
    double m_Altitude {0.0};
    double m_VerticalAccuracy {0.0};
};

}
}