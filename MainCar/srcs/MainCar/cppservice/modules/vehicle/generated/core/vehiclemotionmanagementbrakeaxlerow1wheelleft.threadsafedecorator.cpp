

#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow1wheelleft.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleMotionManagementBrakeAxleRow1WheelLeftThreadSafeDecorator::VehicleMotionManagementBrakeAxleRow1WheelLeftThreadSafeDecorator(std::shared_ptr<IVehicleMotionManagementBrakeAxleRow1WheelLeft> impl)
    : m_impl(impl)
{
}
void VehicleMotionManagementBrakeAxleRow1WheelLeftThreadSafeDecorator::setTorque(int32_t Torque)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMutex);
    m_impl->setTorque(Torque);
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelLeftThreadSafeDecorator::getTorque() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueMutex);
    return m_impl->getTorque();
}
void VehicleMotionManagementBrakeAxleRow1WheelLeftThreadSafeDecorator::setTorqueArbitrated(int32_t TorqueArbitrated)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueArbitratedMutex);
    m_impl->setTorqueArbitrated(TorqueArbitrated);
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelLeftThreadSafeDecorator::getTorqueArbitrated() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueArbitratedMutex);
    return m_impl->getTorqueArbitrated();
}
void VehicleMotionManagementBrakeAxleRow1WheelLeftThreadSafeDecorator::setTorqueFrictionMaximum(int32_t TorqueFrictionMaximum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionMaximumMutex);
    m_impl->setTorqueFrictionMaximum(TorqueFrictionMaximum);
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelLeftThreadSafeDecorator::getTorqueFrictionMaximum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueFrictionMaximumMutex);
    return m_impl->getTorqueFrictionMaximum();
}
void VehicleMotionManagementBrakeAxleRow1WheelLeftThreadSafeDecorator::setTorqueFrictionMinimum(int32_t TorqueFrictionMinimum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionMinimumMutex);
    m_impl->setTorqueFrictionMinimum(TorqueFrictionMinimum);
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelLeftThreadSafeDecorator::getTorqueFrictionMinimum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueFrictionMinimumMutex);
    return m_impl->getTorqueFrictionMinimum();
}
void VehicleMotionManagementBrakeAxleRow1WheelLeftThreadSafeDecorator::setOmegaUpper(int32_t OmegaUpper)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_omegaUpperMutex);
    m_impl->setOmegaUpper(OmegaUpper);
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelLeftThreadSafeDecorator::getOmegaUpper() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_omegaUpperMutex);
    return m_impl->getOmegaUpper();
}
void VehicleMotionManagementBrakeAxleRow1WheelLeftThreadSafeDecorator::setOmegaLower(int32_t OmegaLower)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_omegaLowerMutex);
    m_impl->setOmegaLower(OmegaLower);
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelLeftThreadSafeDecorator::getOmegaLower() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_omegaLowerMutex);
    return m_impl->getOmegaLower();
}

IVehicleMotionManagementBrakeAxleRow1WheelLeftPublisher& VehicleMotionManagementBrakeAxleRow1WheelLeftThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}