

#include "vehicle/generated/core/vehiclebodylights.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleBodyLightsThreadSafeDecorator::VehicleBodyLightsThreadSafeDecorator(std::shared_ptr<IVehicleBodyLights> impl)
    : m_impl(impl)
{
}
void VehicleBodyLightsThreadSafeDecorator::setLightSwitch(Vehicle_Body_Lights_LightSwitch_ValueEnum LightSwitch)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_lightSwitchMutex);
    m_impl->setLightSwitch(LightSwitch);
}

Vehicle_Body_Lights_LightSwitch_ValueEnum VehicleBodyLightsThreadSafeDecorator::getLightSwitch() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_lightSwitchMutex);
    return m_impl->getLightSwitch();
}
void VehicleBodyLightsThreadSafeDecorator::setIsHighBeamSwitchOn(bool IsHighBeamSwitchOn)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isHighBeamSwitchOnMutex);
    m_impl->setIsHighBeamSwitchOn(IsHighBeamSwitchOn);
}

bool VehicleBodyLightsThreadSafeDecorator::getIsHighBeamSwitchOn() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isHighBeamSwitchOnMutex);
    return m_impl->getIsHighBeamSwitchOn();
}

IVehicleBodyLightsPublisher& VehicleBodyLightsThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}