

#include "vehicle/generated/core/vehiclemotionmanagementbrake.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleMotionManagementBrakeThreadSafeDecorator::VehicleMotionManagementBrakeThreadSafeDecorator(std::shared_ptr<IVehicleMotionManagementBrake> impl)
    : m_impl(impl)
{
}
void VehicleMotionManagementBrakeThreadSafeDecorator::setVehicleForceMaximum(int32_t VehicleForceMaximum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleForceMaximumMutex);
    m_impl->setVehicleForceMaximum(VehicleForceMaximum);
}

int32_t VehicleMotionManagementBrakeThreadSafeDecorator::getVehicleForceMaximum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_vehicleForceMaximumMutex);
    return m_impl->getVehicleForceMaximum();
}
void VehicleMotionManagementBrakeThreadSafeDecorator::setVehicleForceElectricMinimumArbitrated(int32_t VehicleForceElectricMinimumArbitrated)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleForceElectricMinimumArbitratedMutex);
    m_impl->setVehicleForceElectricMinimumArbitrated(VehicleForceElectricMinimumArbitrated);
}

int32_t VehicleMotionManagementBrakeThreadSafeDecorator::getVehicleForceElectricMinimumArbitrated() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_vehicleForceElectricMinimumArbitratedMutex);
    return m_impl->getVehicleForceElectricMinimumArbitrated();
}
void VehicleMotionManagementBrakeThreadSafeDecorator::setVehicleForceElectric(int32_t VehicleForceElectric)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleForceElectricMutex);
    m_impl->setVehicleForceElectric(VehicleForceElectric);
}

int32_t VehicleMotionManagementBrakeThreadSafeDecorator::getVehicleForceElectric() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_vehicleForceElectricMutex);
    return m_impl->getVehicleForceElectric();
}
void VehicleMotionManagementBrakeThreadSafeDecorator::setVehicleForceDistributionFrontMaximum(int32_t VehicleForceDistributionFrontMaximum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleForceDistributionFrontMaximumMutex);
    m_impl->setVehicleForceDistributionFrontMaximum(VehicleForceDistributionFrontMaximum);
}

int32_t VehicleMotionManagementBrakeThreadSafeDecorator::getVehicleForceDistributionFrontMaximum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_vehicleForceDistributionFrontMaximumMutex);
    return m_impl->getVehicleForceDistributionFrontMaximum();
}
void VehicleMotionManagementBrakeThreadSafeDecorator::setVehicleForceDistributionFrontMinimum(int32_t VehicleForceDistributionFrontMinimum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleForceDistributionFrontMinimumMutex);
    m_impl->setVehicleForceDistributionFrontMinimum(VehicleForceDistributionFrontMinimum);
}

int32_t VehicleMotionManagementBrakeThreadSafeDecorator::getVehicleForceDistributionFrontMinimum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_vehicleForceDistributionFrontMinimumMutex);
    return m_impl->getVehicleForceDistributionFrontMinimum();
}

IVehicleMotionManagementBrakePublisher& VehicleMotionManagementBrakeThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}