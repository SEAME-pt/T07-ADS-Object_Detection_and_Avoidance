

#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow2wheelright.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator::VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator(std::shared_ptr<IVehicleMotionManagementBrakeAxleRow2WheelRight> impl)
    : m_impl(impl)
{
}
void VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator::setTorque(int32_t Torque)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMutex);
    m_impl->setTorque(Torque);
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator::getTorque() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueMutex);
    return m_impl->getTorque();
}
void VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator::setTorqueArbitrated(int32_t TorqueArbitrated)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueArbitratedMutex);
    m_impl->setTorqueArbitrated(TorqueArbitrated);
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator::getTorqueArbitrated() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueArbitratedMutex);
    return m_impl->getTorqueArbitrated();
}
void VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator::setTorqueFrictionMaximum(int32_t TorqueFrictionMaximum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionMaximumMutex);
    m_impl->setTorqueFrictionMaximum(TorqueFrictionMaximum);
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator::getTorqueFrictionMaximum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueFrictionMaximumMutex);
    return m_impl->getTorqueFrictionMaximum();
}
void VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator::setTorqueFrictionMinimum(int32_t TorqueFrictionMinimum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionMinimumMutex);
    m_impl->setTorqueFrictionMinimum(TorqueFrictionMinimum);
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator::getTorqueFrictionMinimum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueFrictionMinimumMutex);
    return m_impl->getTorqueFrictionMinimum();
}
void VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator::setOmegaUpper(int32_t OmegaUpper)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_omegaUpperMutex);
    m_impl->setOmegaUpper(OmegaUpper);
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator::getOmegaUpper() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_omegaUpperMutex);
    return m_impl->getOmegaUpper();
}
void VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator::setOmegaLower(int32_t OmegaLower)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_omegaLowerMutex);
    m_impl->setOmegaLower(OmegaLower);
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator::getOmegaLower() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_omegaLowerMutex);
    return m_impl->getOmegaLower();
}

IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher& VehicleMotionManagementBrakeAxleRow2WheelRightThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}