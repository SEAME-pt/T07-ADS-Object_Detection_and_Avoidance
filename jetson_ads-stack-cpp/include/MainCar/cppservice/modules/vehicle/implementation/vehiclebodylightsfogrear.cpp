

#include "vehicle/implementation/vehiclebodylightsfogrear.h"
#include "vehicle/generated/core/vehiclebodylightsfogrear.publisher.h"
#include "vehicle/generated/core/vehiclebodylightsfogrear.data.h"

using namespace Cpp::Vehicle;

VehicleBodyLightsFogRear::VehicleBodyLightsFogRear()
    : m_publisher(std::make_unique<VehicleBodyLightsFogRearPublisher>())
{
}
VehicleBodyLightsFogRear::~VehicleBodyLightsFogRear()
{
}

void VehicleBodyLightsFogRear::setIsOn(bool IsOn)
{
    if (m_data.m_IsOn != IsOn) {
        m_data.m_IsOn = IsOn;
        m_publisher->publishIsOnChanged(IsOn);
    }
}

bool VehicleBodyLightsFogRear::getIsOn() const
{
    return m_data.m_IsOn;
}

void VehicleBodyLightsFogRear::setIsDefect(bool IsDefect)
{
    if (m_data.m_IsDefect != IsDefect) {
        m_data.m_IsDefect = IsDefect;
        m_publisher->publishIsDefectChanged(IsDefect);
    }
}

bool VehicleBodyLightsFogRear::getIsDefect() const
{
    return m_data.m_IsDefect;
}

IVehicleBodyLightsFogRearPublisher& VehicleBodyLightsFogRear::_getPublisher() const
{
    return *m_publisher;
}
