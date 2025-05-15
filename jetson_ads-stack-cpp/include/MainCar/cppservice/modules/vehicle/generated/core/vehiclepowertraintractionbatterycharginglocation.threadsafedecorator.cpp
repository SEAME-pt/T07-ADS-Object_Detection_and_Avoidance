

#include "vehicle/generated/core/vehiclepowertraintractionbatterycharginglocation.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehiclePowertrainTractionBatteryChargingLocationThreadSafeDecorator::VehiclePowertrainTractionBatteryChargingLocationThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryChargingLocation> impl)
    : m_impl(impl)
{
}
void VehiclePowertrainTractionBatteryChargingLocationThreadSafeDecorator::setLatitude(double Latitude)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_latitudeMutex);
    m_impl->setLatitude(Latitude);
}

double VehiclePowertrainTractionBatteryChargingLocationThreadSafeDecorator::getLatitude() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_latitudeMutex);
    return m_impl->getLatitude();
}
void VehiclePowertrainTractionBatteryChargingLocationThreadSafeDecorator::setLongitude(double Longitude)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_longitudeMutex);
    m_impl->setLongitude(Longitude);
}

double VehiclePowertrainTractionBatteryChargingLocationThreadSafeDecorator::getLongitude() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_longitudeMutex);
    return m_impl->getLongitude();
}
void VehiclePowertrainTractionBatteryChargingLocationThreadSafeDecorator::setAltitude(double Altitude)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_altitudeMutex);
    m_impl->setAltitude(Altitude);
}

double VehiclePowertrainTractionBatteryChargingLocationThreadSafeDecorator::getAltitude() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_altitudeMutex);
    return m_impl->getAltitude();
}

IVehiclePowertrainTractionBatteryChargingLocationPublisher& VehiclePowertrainTractionBatteryChargingLocationThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}