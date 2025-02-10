

#include "vehicle/generated/core/vehiclemotionmanagementsteeringsteeringwheel.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator(std::shared_ptr<IVehicleMotionManagementSteeringSteeringWheel> impl)
    : m_impl(impl)
{
}
void VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::setAngle(int32_t Angle)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_angleMutex);
    m_impl->setAngle(Angle);
}

int32_t VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::getAngle() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_angleMutex);
    return m_impl->getAngle();
}
void VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::setAngleTarget(int32_t AngleTarget)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_angleTargetMutex);
    m_impl->setAngleTarget(AngleTarget);
}

int32_t VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::getAngleTarget() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_angleTargetMutex);
    return m_impl->getAngleTarget();
}
void VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::setAngleTargetMode(int32_t AngleTargetMode)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_angleTargetModeMutex);
    m_impl->setAngleTargetMode(AngleTargetMode);
}

int32_t VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::getAngleTargetMode() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_angleTargetModeMutex);
    return m_impl->getAngleTargetMode();
}
void VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::setTorqueOffsetTarget(int32_t TorqueOffsetTarget)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueOffsetTargetMutex);
    m_impl->setTorqueOffsetTarget(TorqueOffsetTarget);
}

int32_t VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::getTorqueOffsetTarget() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueOffsetTargetMutex);
    return m_impl->getTorqueOffsetTarget();
}
void VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::setTorqueOffsetTargetMode(int32_t TorqueOffsetTargetMode)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueOffsetTargetModeMutex);
    m_impl->setTorqueOffsetTargetMode(TorqueOffsetTargetMode);
}

int32_t VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::getTorqueOffsetTargetMode() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueOffsetTargetModeMutex);
    return m_impl->getTorqueOffsetTargetMode();
}
void VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::setTorqueTarget(int32_t TorqueTarget)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueTargetMutex);
    m_impl->setTorqueTarget(TorqueTarget);
}

int32_t VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::getTorqueTarget() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueTargetMutex);
    return m_impl->getTorqueTarget();
}
void VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::setTorqueTargetMode(int32_t TorqueTargetMode)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueTargetModeMutex);
    m_impl->setTorqueTargetMode(TorqueTargetMode);
}

int32_t VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::getTorqueTargetMode() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueTargetModeMutex);
    return m_impl->getTorqueTargetMode();
}
void VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::setTorque(int32_t Torque)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMutex);
    m_impl->setTorque(Torque);
}

int32_t VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::getTorque() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueMutex);
    return m_impl->getTorque();
}

IVehicleMotionManagementSteeringSteeringWheelPublisher& VehicleMotionManagementSteeringSteeringWheelThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}