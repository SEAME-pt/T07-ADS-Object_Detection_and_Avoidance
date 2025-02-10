

#include "vehicle/generated/core/vehicleconnectivity.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleConnectivityThreadSafeDecorator::VehicleConnectivityThreadSafeDecorator(std::shared_ptr<IVehicleConnectivity> impl)
    : m_impl(impl)
{
}
void VehicleConnectivityThreadSafeDecorator::setIsConnectivityAvailable(bool IsConnectivityAvailable)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isConnectivityAvailableMutex);
    m_impl->setIsConnectivityAvailable(IsConnectivityAvailable);
}

bool VehicleConnectivityThreadSafeDecorator::getIsConnectivityAvailable() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_isConnectivityAvailableMutex);
    return m_impl->getIsConnectivityAvailable();
}

IVehicleConnectivityPublisher& VehicleConnectivityThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}