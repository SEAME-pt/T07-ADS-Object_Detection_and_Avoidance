

#include "vehicle/generated/core/vehiclebodylightsbackup.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleBodyLightsBackupThreadSafeDecorator::VehicleBodyLightsBackupThreadSafeDecorator(std::shared_ptr<IVehicleBodyLightsBackup> impl)
    : m_impl(impl)
{
}
void VehicleBodyLightsBackupThreadSafeDecorator::setIsOn(bool IsOn)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isOnMutex);
    m_impl->setIsOn(IsOn);
}

bool VehicleBodyLightsBackupThreadSafeDecorator::getIsOn() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isOnMutex);
    return m_impl->getIsOn();
}
void VehicleBodyLightsBackupThreadSafeDecorator::setIsDefect(bool IsDefect)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    m_impl->setIsDefect(IsDefect);
}

bool VehicleBodyLightsBackupThreadSafeDecorator::getIsDefect() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isDefectMutex);
    return m_impl->getIsDefect();
}

IVehicleBodyLightsBackupPublisher& VehicleBodyLightsBackupThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}