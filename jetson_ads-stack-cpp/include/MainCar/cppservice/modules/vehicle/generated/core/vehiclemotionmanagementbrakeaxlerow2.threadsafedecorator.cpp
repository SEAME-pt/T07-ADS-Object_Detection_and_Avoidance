

#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow2.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleMotionManagementBrakeAxleRow2ThreadSafeDecorator::VehicleMotionManagementBrakeAxleRow2ThreadSafeDecorator(std::shared_ptr<IVehicleMotionManagementBrakeAxleRow2> impl)
    : m_impl(impl)
{
}
void VehicleMotionManagementBrakeAxleRow2ThreadSafeDecorator::setTorqueElectricMinimum(int32_t TorqueElectricMinimum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueElectricMinimumMutex);
    m_impl->setTorqueElectricMinimum(TorqueElectricMinimum);
}

int32_t VehicleMotionManagementBrakeAxleRow2ThreadSafeDecorator::getTorqueElectricMinimum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueElectricMinimumMutex);
    return m_impl->getTorqueElectricMinimum();
}
void VehicleMotionManagementBrakeAxleRow2ThreadSafeDecorator::setTorqueFrictionDifferenceMaximum(int32_t TorqueFrictionDifferenceMaximum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionDifferenceMaximumMutex);
    m_impl->setTorqueFrictionDifferenceMaximum(TorqueFrictionDifferenceMaximum);
}

int32_t VehicleMotionManagementBrakeAxleRow2ThreadSafeDecorator::getTorqueFrictionDifferenceMaximum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueFrictionDifferenceMaximumMutex);
    return m_impl->getTorqueFrictionDifferenceMaximum();
}
void VehicleMotionManagementBrakeAxleRow2ThreadSafeDecorator::setTorqueDistributionFrictionRightMaximum(int32_t TorqueDistributionFrictionRightMaximum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueDistributionFrictionRightMaximumMutex);
    m_impl->setTorqueDistributionFrictionRightMaximum(TorqueDistributionFrictionRightMaximum);
}

int32_t VehicleMotionManagementBrakeAxleRow2ThreadSafeDecorator::getTorqueDistributionFrictionRightMaximum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueDistributionFrictionRightMaximumMutex);
    return m_impl->getTorqueDistributionFrictionRightMaximum();
}
void VehicleMotionManagementBrakeAxleRow2ThreadSafeDecorator::setTorqueDistributionFrictionRightMinimum(int32_t TorqueDistributionFrictionRightMinimum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueDistributionFrictionRightMinimumMutex);
    m_impl->setTorqueDistributionFrictionRightMinimum(TorqueDistributionFrictionRightMinimum);
}

int32_t VehicleMotionManagementBrakeAxleRow2ThreadSafeDecorator::getTorqueDistributionFrictionRightMinimum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueDistributionFrictionRightMinimumMutex);
    return m_impl->getTorqueDistributionFrictionRightMinimum();
}

IVehicleMotionManagementBrakeAxleRow2Publisher& VehicleMotionManagementBrakeAxleRow2ThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}