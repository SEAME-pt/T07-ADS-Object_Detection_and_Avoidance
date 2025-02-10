

#include "vehicle/generated/core/vehiclecurrentlocationgnssreceiver.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleCurrentLocationGnssReceiverThreadSafeDecorator::VehicleCurrentLocationGnssReceiverThreadSafeDecorator(std::shared_ptr<IVehicleCurrentLocationGnssReceiver> impl)
    : m_impl(impl)
{
}
void VehicleCurrentLocationGnssReceiverThreadSafeDecorator::setFixType(Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum FixType)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_fixTypeMutex);
    m_impl->setFixType(FixType);
}

Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum VehicleCurrentLocationGnssReceiverThreadSafeDecorator::getFixType() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_fixTypeMutex);
    return m_impl->getFixType();
}

IVehicleCurrentLocationGnssReceiverPublisher& VehicleCurrentLocationGnssReceiverThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}