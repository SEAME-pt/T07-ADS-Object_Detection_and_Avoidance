

#include "vehicle/implementation/vehiclebodylightsparking.h"
#include "vehicle/generated/core/vehiclebodylightsparking.publisher.h"
#include "vehicle/generated/core/vehiclebodylightsparking.data.h"

using namespace Cpp::Vehicle;

VehicleBodyLightsParking::VehicleBodyLightsParking()
    : m_publisher(std::make_unique<VehicleBodyLightsParkingPublisher>())
{
}
VehicleBodyLightsParking::~VehicleBodyLightsParking()
{
}

void VehicleBodyLightsParking::setIsOn(bool IsOn)
{
    if (m_data.m_IsOn != IsOn) {
        m_data.m_IsOn = IsOn;
        m_publisher->publishIsOnChanged(IsOn);
    }
}

bool VehicleBodyLightsParking::getIsOn() const
{
    return m_data.m_IsOn;
}

void VehicleBodyLightsParking::setIsDefect(bool IsDefect)
{
    if (m_data.m_IsDefect != IsDefect) {
        m_data.m_IsDefect = IsDefect;
        m_publisher->publishIsDefectChanged(IsDefect);
    }
}

bool VehicleBodyLightsParking::getIsDefect() const
{
    return m_data.m_IsDefect;
}

IVehicleBodyLightsParkingPublisher& VehicleBodyLightsParking::_getPublisher() const
{
    return *m_publisher;
}
