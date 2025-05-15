

#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow1.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleMotionManagementBrakeAxleRow1ThreadSafeDecorator::VehicleMotionManagementBrakeAxleRow1ThreadSafeDecorator(std::shared_ptr<IVehicleMotionManagementBrakeAxleRow1> impl)
    : m_impl(impl)
{
}
void VehicleMotionManagementBrakeAxleRow1ThreadSafeDecorator::setTorqueElectricMinimum(int32_t TorqueElectricMinimum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueElectricMinimumMutex);
    m_impl->setTorqueElectricMinimum(TorqueElectricMinimum);
}

int32_t VehicleMotionManagementBrakeAxleRow1ThreadSafeDecorator::getTorqueElectricMinimum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueElectricMinimumMutex);
    return m_impl->getTorqueElectricMinimum();
}
void VehicleMotionManagementBrakeAxleRow1ThreadSafeDecorator::setTorqueFrictionDifferenceMaximum(int32_t TorqueFrictionDifferenceMaximum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionDifferenceMaximumMutex);
    m_impl->setTorqueFrictionDifferenceMaximum(TorqueFrictionDifferenceMaximum);
}

int32_t VehicleMotionManagementBrakeAxleRow1ThreadSafeDecorator::getTorqueFrictionDifferenceMaximum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueFrictionDifferenceMaximumMutex);
    return m_impl->getTorqueFrictionDifferenceMaximum();
}
void VehicleMotionManagementBrakeAxleRow1ThreadSafeDecorator::setTorqueDistributionFrictionRightMaximum(int32_t TorqueDistributionFrictionRightMaximum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueDistributionFrictionRightMaximumMutex);
    m_impl->setTorqueDistributionFrictionRightMaximum(TorqueDistributionFrictionRightMaximum);
}

int32_t VehicleMotionManagementBrakeAxleRow1ThreadSafeDecorator::getTorqueDistributionFrictionRightMaximum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueDistributionFrictionRightMaximumMutex);
    return m_impl->getTorqueDistributionFrictionRightMaximum();
}
void VehicleMotionManagementBrakeAxleRow1ThreadSafeDecorator::setTorqueDistributionFrictionRightMinimum(int32_t TorqueDistributionFrictionRightMinimum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueDistributionFrictionRightMinimumMutex);
    m_impl->setTorqueDistributionFrictionRightMinimum(TorqueDistributionFrictionRightMinimum);
}

int32_t VehicleMotionManagementBrakeAxleRow1ThreadSafeDecorator::getTorqueDistributionFrictionRightMinimum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueDistributionFrictionRightMinimumMutex);
    return m_impl->getTorqueDistributionFrictionRightMinimum();
}

IVehicleMotionManagementBrakeAxleRow1Publisher& VehicleMotionManagementBrakeAxleRow1ThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}