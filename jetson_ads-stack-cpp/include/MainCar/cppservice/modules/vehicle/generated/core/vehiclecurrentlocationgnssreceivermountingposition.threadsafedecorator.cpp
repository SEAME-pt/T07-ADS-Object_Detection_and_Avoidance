

#include "vehicle/generated/core/vehiclecurrentlocationgnssreceivermountingposition.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleCurrentLocationGnssReceiverMountingPositionThreadSafeDecorator::VehicleCurrentLocationGnssReceiverMountingPositionThreadSafeDecorator(std::shared_ptr<IVehicleCurrentLocationGnssReceiverMountingPosition> impl)
    : m_impl(impl)
{
}
void VehicleCurrentLocationGnssReceiverMountingPositionThreadSafeDecorator::setX(int32_t X)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_xMutex);
    m_impl->setX(X);
}

int32_t VehicleCurrentLocationGnssReceiverMountingPositionThreadSafeDecorator::getX() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_xMutex);
    return m_impl->getX();
}
void VehicleCurrentLocationGnssReceiverMountingPositionThreadSafeDecorator::setY(int32_t Y)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_yMutex);
    m_impl->setY(Y);
}

int32_t VehicleCurrentLocationGnssReceiverMountingPositionThreadSafeDecorator::getY() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_yMutex);
    return m_impl->getY();
}
void VehicleCurrentLocationGnssReceiverMountingPositionThreadSafeDecorator::setZ(int32_t Z)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_zMutex);
    m_impl->setZ(Z);
}

int32_t VehicleCurrentLocationGnssReceiverMountingPositionThreadSafeDecorator::getZ() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_zMutex);
    return m_impl->getZ();
}

IVehicleCurrentLocationGnssReceiverMountingPositionPublisher& VehicleCurrentLocationGnssReceiverMountingPositionThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}