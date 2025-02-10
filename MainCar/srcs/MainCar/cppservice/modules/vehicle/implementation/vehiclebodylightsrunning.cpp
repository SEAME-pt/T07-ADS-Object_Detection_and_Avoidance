

#include "vehicle/implementation/vehiclebodylightsrunning.h"
#include "vehicle/generated/core/vehiclebodylightsrunning.publisher.h"
#include "vehicle/generated/core/vehiclebodylightsrunning.data.h"

using namespace Cpp::Vehicle;

VehicleBodyLightsRunning::VehicleBodyLightsRunning()
    : m_publisher(std::make_unique<VehicleBodyLightsRunningPublisher>())
{
}
VehicleBodyLightsRunning::~VehicleBodyLightsRunning()
{
}

void VehicleBodyLightsRunning::setIsOn(bool IsOn)
{
    if (m_data.m_IsOn != IsOn) {
        m_data.m_IsOn = IsOn;
        m_publisher->publishIsOnChanged(IsOn);
    }
}

bool VehicleBodyLightsRunning::getIsOn() const
{
    return m_data.m_IsOn;
}

void VehicleBodyLightsRunning::setIsDefect(bool IsDefect)
{
    if (m_data.m_IsDefect != IsDefect) {
        m_data.m_IsDefect = IsDefect;
        m_publisher->publishIsDefectChanged(IsDefect);
    }
}

bool VehicleBodyLightsRunning::getIsDefect() const
{
    return m_data.m_IsDefect;
}

IVehicleBodyLightsRunningPublisher& VehicleBodyLightsRunning::_getPublisher() const
{
    return *m_publisher;
}
