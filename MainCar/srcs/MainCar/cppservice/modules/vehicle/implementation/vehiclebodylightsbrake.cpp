

#include "vehicle/implementation/vehiclebodylightsbrake.h"
#include "vehicle/generated/core/vehiclebodylightsbrake.publisher.h"
#include "vehicle/generated/core/vehiclebodylightsbrake.data.h"

using namespace Cpp::Vehicle;

VehicleBodyLightsBrake::VehicleBodyLightsBrake()
    : m_publisher(std::make_unique<VehicleBodyLightsBrakePublisher>())
{
}
VehicleBodyLightsBrake::~VehicleBodyLightsBrake()
{
}

void VehicleBodyLightsBrake::setIsActive(Vehicle_Body_Lights_Brake_IsActive_ValueEnum IsActive)
{
    if (m_data.m_IsActive != IsActive) {
        m_data.m_IsActive = IsActive;
        m_publisher->publishIsActiveChanged(IsActive);
    }
}

Vehicle_Body_Lights_Brake_IsActive_ValueEnum VehicleBodyLightsBrake::getIsActive() const
{
    return m_data.m_IsActive;
}

void VehicleBodyLightsBrake::setIsDefect(bool IsDefect)
{
    if (m_data.m_IsDefect != IsDefect) {
        m_data.m_IsDefect = IsDefect;
        m_publisher->publishIsDefectChanged(IsDefect);
    }
}

bool VehicleBodyLightsBrake::getIsDefect() const
{
    return m_data.m_IsDefect;
}

IVehicleBodyLightsBrakePublisher& VehicleBodyLightsBrake::_getPublisher() const
{
    return *m_publisher;
}
