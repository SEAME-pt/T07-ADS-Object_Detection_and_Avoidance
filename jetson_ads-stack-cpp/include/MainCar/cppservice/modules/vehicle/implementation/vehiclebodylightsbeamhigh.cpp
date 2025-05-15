

#include "vehicle/implementation/vehiclebodylightsbeamhigh.h"
#include "vehicle/generated/core/vehiclebodylightsbeamhigh.publisher.h"
#include "vehicle/generated/core/vehiclebodylightsbeamhigh.data.h"

using namespace Cpp::Vehicle;

VehicleBodyLightsBeamHigh::VehicleBodyLightsBeamHigh()
    : m_publisher(std::make_unique<VehicleBodyLightsBeamHighPublisher>())
{
}
VehicleBodyLightsBeamHigh::~VehicleBodyLightsBeamHigh()
{
}

void VehicleBodyLightsBeamHigh::setIsOn(bool IsOn)
{
    if (m_data.m_IsOn != IsOn) {
        m_data.m_IsOn = IsOn;
        m_publisher->publishIsOnChanged(IsOn);
    }
}

bool VehicleBodyLightsBeamHigh::getIsOn() const
{
    return m_data.m_IsOn;
}

void VehicleBodyLightsBeamHigh::setIsDefect(bool IsDefect)
{
    if (m_data.m_IsDefect != IsDefect) {
        m_data.m_IsDefect = IsDefect;
        m_publisher->publishIsDefectChanged(IsDefect);
    }
}

bool VehicleBodyLightsBeamHigh::getIsDefect() const
{
    return m_data.m_IsDefect;
}

IVehicleBodyLightsBeamHighPublisher& VehicleBodyLightsBeamHigh::_getPublisher() const
{
    return *m_publisher;
}
