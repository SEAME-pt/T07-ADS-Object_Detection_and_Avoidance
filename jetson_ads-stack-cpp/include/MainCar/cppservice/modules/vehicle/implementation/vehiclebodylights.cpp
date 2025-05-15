

#include "vehicle/implementation/vehiclebodylights.h"
#include "vehicle/generated/core/vehiclebodylights.publisher.h"
#include "vehicle/generated/core/vehiclebodylights.data.h"

using namespace Cpp::Vehicle;

VehicleBodyLights::VehicleBodyLights()
    : m_publisher(std::make_unique<VehicleBodyLightsPublisher>())
{
}
VehicleBodyLights::~VehicleBodyLights()
{
}

void VehicleBodyLights::setLightSwitch(Vehicle_Body_Lights_LightSwitch_ValueEnum LightSwitch)
{
    if (m_data.m_LightSwitch != LightSwitch) {
        m_data.m_LightSwitch = LightSwitch;
        m_publisher->publishLightSwitchChanged(LightSwitch);
    }
}

Vehicle_Body_Lights_LightSwitch_ValueEnum VehicleBodyLights::getLightSwitch() const
{
    return m_data.m_LightSwitch;
}

void VehicleBodyLights::setIsHighBeamSwitchOn(bool IsHighBeamSwitchOn)
{
    if (m_data.m_IsHighBeamSwitchOn != IsHighBeamSwitchOn) {
        m_data.m_IsHighBeamSwitchOn = IsHighBeamSwitchOn;
        m_publisher->publishIsHighBeamSwitchOnChanged(IsHighBeamSwitchOn);
    }
}

bool VehicleBodyLights::getIsHighBeamSwitchOn() const
{
    return m_data.m_IsHighBeamSwitchOn;
}

IVehicleBodyLightsPublisher& VehicleBodyLights::_getPublisher() const
{
    return *m_publisher;
}
