

#include "vehicle/generated/core/vehiclebodylightsbrake.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleBodyLightsBrakeThreadSafeDecorator::VehicleBodyLightsBrakeThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsBrake> impl)
    : m_impl(impl)
{
}
void VehicleBodyLightsBrakeThreadSafeDecorator::setIsActive(Vehicle_Body_Lights_Brake_IsActive_ValueEnum IsActive)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isActiveMutex);
    m_impl->setIsActive(IsActive);
}

Vehicle_Body_Lights_Brake_IsActive_ValueEnum VehicleBodyLightsBrakeThreadSafeDecorator::getIsActive() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isActiveMutex);
    return m_impl->getIsActive();
}
void VehicleBodyLightsBrakeThreadSafeDecorator::setIsDefect(bool IsDefect)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    m_impl->setIsDefect(IsDefect);
}

bool VehicleBodyLightsBrakeThreadSafeDecorator::getIsDefect() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    return m_impl->getIsDefect();
}

IVehicleBodyLightsBrakePublisher& VehicleBodyLightsBrakeThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}