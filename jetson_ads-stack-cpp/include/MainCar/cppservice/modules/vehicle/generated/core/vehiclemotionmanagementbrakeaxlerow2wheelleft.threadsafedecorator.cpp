

#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow2wheelleft.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleMotionManagementBrakeAxleRow2WheelLeftThreadSafeDecorator::VehicleMotionManagementBrakeAxleRow2WheelLeftThreadSafeDecorator(std::shared_ptr<IVehicleMotionManagementBrakeAxleRow2WheelLeft> impl)
    : m_impl(impl)
{
}
void VehicleMotionManagementBrakeAxleRow2WheelLeftThreadSafeDecorator::setTorque(int32_t Torque)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMutex);
    m_impl->setTorque(Torque);
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelLeftThreadSafeDecorator::getTorque() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueMutex);
    return m_impl->getTorque();
}
void VehicleMotionManagementBrakeAxleRow2WheelLeftThreadSafeDecorator::setTorqueArbitrated(int32_t TorqueArbitrated)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueArbitratedMutex);
    m_impl->setTorqueArbitrated(TorqueArbitrated);
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelLeftThreadSafeDecorator::getTorqueArbitrated() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueArbitratedMutex);
    return m_impl->getTorqueArbitrated();
}
void VehicleMotionManagementBrakeAxleRow2WheelLeftThreadSafeDecorator::setTorqueFrictionMaximum(int32_t TorqueFrictionMaximum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionMaximumMutex);
    m_impl->setTorqueFrictionMaximum(TorqueFrictionMaximum);
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelLeftThreadSafeDecorator::getTorqueFrictionMaximum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueFrictionMaximumMutex);
    return m_impl->getTorqueFrictionMaximum();
}
void VehicleMotionManagementBrakeAxleRow2WheelLeftThreadSafeDecorator::setTorqueFrictionMinimum(int32_t TorqueFrictionMinimum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionMinimumMutex);
    m_impl->setTorqueFrictionMinimum(TorqueFrictionMinimum);
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelLeftThreadSafeDecorator::getTorqueFrictionMinimum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueFrictionMinimumMutex);
    return m_impl->getTorqueFrictionMinimum();
}
void VehicleMotionManagementBrakeAxleRow2WheelLeftThreadSafeDecorator::setOmegaUpper(int32_t OmegaUpper)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_omegaUpperMutex);
    m_impl->setOmegaUpper(OmegaUpper);
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelLeftThreadSafeDecorator::getOmegaUpper() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_omegaUpperMutex);
    return m_impl->getOmegaUpper();
}
void VehicleMotionManagementBrakeAxleRow2WheelLeftThreadSafeDecorator::setOmegaLower(int32_t OmegaLower)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_omegaLowerMutex);
    m_impl->setOmegaLower(OmegaLower);
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelLeftThreadSafeDecorator::getOmegaLower() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_omegaLowerMutex);
    return m_impl->getOmegaLower();
}

IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher& VehicleMotionManagementBrakeAxleRow2WheelLeftThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}