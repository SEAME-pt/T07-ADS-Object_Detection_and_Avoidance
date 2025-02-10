

#include "vehicle/implementation/vehiclebodylightsfogfront.h"
#include "vehicle/generated/core/vehiclebodylightsfogfront.publisher.h"
#include "vehicle/generated/core/vehiclebodylightsfogfront.data.h"

using namespace Cpp::Vehicle;

VehicleBodyLightsFogFront::VehicleBodyLightsFogFront()
    : m_publisher(std::make_unique<VehicleBodyLightsFogFrontPublisher>())
{
}
VehicleBodyLightsFogFront::~VehicleBodyLightsFogFront()
{
}

void VehicleBodyLightsFogFront::setIsOn(bool IsOn)
{
    if (m_data.m_IsOn != IsOn) {
        m_data.m_IsOn = IsOn;
        m_publisher->publishIsOnChanged(IsOn);
    }
}

bool VehicleBodyLightsFogFront::getIsOn() const
{
    return m_data.m_IsOn;
}

void VehicleBodyLightsFogFront::setIsDefect(bool IsDefect)
{
    if (m_data.m_IsDefect != IsDefect) {
        m_data.m_IsDefect = IsDefect;
        m_publisher->publishIsDefectChanged(IsDefect);
    }
}

bool VehicleBodyLightsFogFront::getIsDefect() const
{
    return m_data.m_IsDefect;
}

IVehicleBodyLightsFogFrontPublisher& VehicleBodyLightsFogFront::_getPublisher() const
{
    return *m_publisher;
}
