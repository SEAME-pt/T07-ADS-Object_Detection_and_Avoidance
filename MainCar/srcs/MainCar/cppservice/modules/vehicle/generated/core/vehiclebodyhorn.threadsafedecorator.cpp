

#include "vehicle/generated/core/vehiclebodyhorn.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleBodyHornThreadSafeDecorator::VehicleBodyHornThreadSafeDecorator(std::shared_ptr<IVehicleBodyHorn> impl)
    : m_impl(impl)
{
}
void VehicleBodyHornThreadSafeDecorator::setIsActive(bool IsActive)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isActiveMutex);
    m_impl->setIsActive(IsActive);
}

bool VehicleBodyHornThreadSafeDecorator::getIsActive() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isActiveMutex);
    return m_impl->getIsActive();
}

IVehicleBodyHornPublisher& VehicleBodyHornThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}