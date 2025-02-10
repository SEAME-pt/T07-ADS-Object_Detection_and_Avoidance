

#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow1wheelright.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator::VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator(std::shared_ptr<IVehicleMotionManagementBrakeAxleRow1WheelRight> impl)
    : m_impl(impl)
{
}
void VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator::setTorque(int32_t Torque)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMutex);
    m_impl->setTorque(Torque);
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator::getTorque() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueMutex);
    return m_impl->getTorque();
}
void VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator::setTorqueArbitrated(int32_t TorqueArbitrated)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueArbitratedMutex);
    m_impl->setTorqueArbitrated(TorqueArbitrated);
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator::getTorqueArbitrated() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueArbitratedMutex);
    return m_impl->getTorqueArbitrated();
}
void VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator::setTorqueFrictionMaximum(int32_t TorqueFrictionMaximum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionMaximumMutex);
    m_impl->setTorqueFrictionMaximum(TorqueFrictionMaximum);
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator::getTorqueFrictionMaximum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueFrictionMaximumMutex);
    return m_impl->getTorqueFrictionMaximum();
}
void VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator::setTorqueFrictionMinimum(int32_t TorqueFrictionMinimum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionMinimumMutex);
    m_impl->setTorqueFrictionMinimum(TorqueFrictionMinimum);
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator::getTorqueFrictionMinimum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueFrictionMinimumMutex);
    return m_impl->getTorqueFrictionMinimum();
}
void VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator::setOmegaUpper(int32_t OmegaUpper)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_omegaUpperMutex);
    m_impl->setOmegaUpper(OmegaUpper);
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator::getOmegaUpper() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_omegaUpperMutex);
    return m_impl->getOmegaUpper();
}
void VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator::setOmegaLower(int32_t OmegaLower)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_omegaLowerMutex);
    m_impl->setOmegaLower(OmegaLower);
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator::getOmegaLower() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_omegaLowerMutex);
    return m_impl->getOmegaLower();
}

IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher& VehicleMotionManagementBrakeAxleRow1WheelRightThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}