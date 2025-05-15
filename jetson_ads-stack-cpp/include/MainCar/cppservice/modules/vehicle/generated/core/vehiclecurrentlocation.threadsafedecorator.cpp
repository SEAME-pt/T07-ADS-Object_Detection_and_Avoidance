

#include "vehicle/generated/core/vehiclecurrentlocation.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleCurrentLocationThreadSafeDecorator::VehicleCurrentLocationThreadSafeDecorator(std::shared_ptr<IVehicleCurrentLocation> impl)
    : m_impl(impl)
{
}
void VehicleCurrentLocationThreadSafeDecorator::setTimestamp(const std::string& Timestamp)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_timestampMutex);
    m_impl->setTimestamp(Timestamp);
}

const std::string& VehicleCurrentLocationThreadSafeDecorator::getTimestamp() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_timestampMutex);
    return m_impl->getTimestamp();
}
void VehicleCurrentLocationThreadSafeDecorator::setLatitude(double Latitude)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_latitudeMutex);
    m_impl->setLatitude(Latitude);
}

double VehicleCurrentLocationThreadSafeDecorator::getLatitude() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_latitudeMutex);
    return m_impl->getLatitude();
}
void VehicleCurrentLocationThreadSafeDecorator::setLongitude(double Longitude)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_longitudeMutex);
    m_impl->setLongitude(Longitude);
}

double VehicleCurrentLocationThreadSafeDecorator::getLongitude() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_longitudeMutex);
    return m_impl->getLongitude();
}
void VehicleCurrentLocationThreadSafeDecorator::setHeading(double Heading)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_headingMutex);
    m_impl->setHeading(Heading);
}

double VehicleCurrentLocationThreadSafeDecorator::getHeading() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_headingMutex);
    return m_impl->getHeading();
}
void VehicleCurrentLocationThreadSafeDecorator::setHorizontalAccuracy(double HorizontalAccuracy)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_horizontalAccuracyMutex);
    m_impl->setHorizontalAccuracy(HorizontalAccuracy);
}

double VehicleCurrentLocationThreadSafeDecorator::getHorizontalAccuracy() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_horizontalAccuracyMutex);
    return m_impl->getHorizontalAccuracy();
}
void VehicleCurrentLocationThreadSafeDecorator::setAltitude(double Altitude)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_altitudeMutex);
    m_impl->setAltitude(Altitude);
}

double VehicleCurrentLocationThreadSafeDecorator::getAltitude() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_altitudeMutex);
    return m_impl->getAltitude();
}
void VehicleCurrentLocationThreadSafeDecorator::setVerticalAccuracy(double VerticalAccuracy)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_verticalAccuracyMutex);
    m_impl->setVerticalAccuracy(VerticalAccuracy);
}

double VehicleCurrentLocationThreadSafeDecorator::getVerticalAccuracy() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_verticalAccuracyMutex);
    return m_impl->getVerticalAccuracy();
}

IVehicleCurrentLocationPublisher& VehicleCurrentLocationThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}