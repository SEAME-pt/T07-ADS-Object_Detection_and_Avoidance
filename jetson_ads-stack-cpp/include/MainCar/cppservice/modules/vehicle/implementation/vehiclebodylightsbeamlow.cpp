

#include "vehicle/implementation/vehiclebodylightsbeamlow.h"
#include "vehicle/generated/core/vehiclebodylightsbeamlow.publisher.h"
#include "vehicle/generated/core/vehiclebodylightsbeamlow.data.h"

using namespace Cpp::Vehicle;

VehicleBodyLightsBeamLow::VehicleBodyLightsBeamLow()
    : m_publisher(std::make_unique<VehicleBodyLightsBeamLowPublisher>())
{
}
VehicleBodyLightsBeamLow::~VehicleBodyLightsBeamLow()
{
}

void VehicleBodyLightsBeamLow::setIsOn(bool IsOn)
{
    if (m_data.m_IsOn != IsOn) {
        m_data.m_IsOn = IsOn;
        m_publisher->publishIsOnChanged(IsOn);
    }
}

bool VehicleBodyLightsBeamLow::getIsOn() const
{
    return m_data.m_IsOn;
}

void VehicleBodyLightsBeamLow::setIsDefect(bool IsDefect)
{
    if (m_data.m_IsDefect != IsDefect) {
        m_data.m_IsDefect = IsDefect;
        m_publisher->publishIsDefectChanged(IsDefect);
    }
}

bool VehicleBodyLightsBeamLow::getIsDefect() const
{
    return m_data.m_IsDefect;
}

IVehicleBodyLightsBeamLowPublisher& VehicleBodyLightsBeamLow::_getPublisher() const
{
    return *m_publisher;
}
