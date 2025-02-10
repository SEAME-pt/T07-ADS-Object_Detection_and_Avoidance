#pragma once

#include "vehicle/generated/api/vehicle.h"


namespace Cpp
{
namespace Vehicle
{

/**
* A helper structure for implementations of Vehicle_Driver_Identifier_Deprecated. Stores all the properties.
*/
struct VehicleDriverIdentifierDeprecatedData
{
    std::string m_Subject_Deprecated {std::string()};
    std::string m_Issuer_Deprecated {std::string()};
};

}
}