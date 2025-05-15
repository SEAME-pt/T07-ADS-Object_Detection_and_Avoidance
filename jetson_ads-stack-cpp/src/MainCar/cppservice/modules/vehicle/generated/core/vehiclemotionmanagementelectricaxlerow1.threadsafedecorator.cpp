

#include "vehicle/generated/core/vehiclemotionmanagementelectricaxlerow1.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator(std::shared_ptr<IVehicleMotionManagementElectricAxleRow1> impl)
    : m_impl(impl)
{
}
void VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::setTorque(int32_t Torque)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMutex);
    m_impl->setTorque(Torque);
}

int32_t VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::getTorque() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueMutex);
    return m_impl->getTorque();
}
void VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::setTorqueTarget(int32_t TorqueTarget)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueTargetMutex);
    m_impl->setTorqueTarget(TorqueTarget);
}

int32_t VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::getTorqueTarget() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueTargetMutex);
    return m_impl->getTorqueTarget();
}
void VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::setTorqueMaximum(int32_t TorqueMaximum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMaximumMutex);
    m_impl->setTorqueMaximum(TorqueMaximum);
}

int32_t VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::getTorqueMaximum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueMaximumMutex);
    return m_impl->getTorqueMaximum();
}
void VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::setTorqueMinimum(int32_t TorqueMinimum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMinimumMutex);
    m_impl->setTorqueMinimum(TorqueMinimum);
}

int32_t VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::getTorqueMinimum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueMinimumMutex);
    return m_impl->getTorqueMinimum();
}
void VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::setTorqueLongTermMaximum(int32_t TorqueLongTermMaximum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueLongTermMaximumMutex);
    m_impl->setTorqueLongTermMaximum(TorqueLongTermMaximum);
}

int32_t VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::getTorqueLongTermMaximum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueLongTermMaximumMutex);
    return m_impl->getTorqueLongTermMaximum();
}
void VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::setTorqueLongTermMinimum(int32_t TorqueLongTermMinimum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueLongTermMinimumMutex);
    m_impl->setTorqueLongTermMinimum(TorqueLongTermMinimum);
}

int32_t VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::getTorqueLongTermMinimum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueLongTermMinimumMutex);
    return m_impl->getTorqueLongTermMinimum();
}
void VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::setTorqueShortTermMaximum(int32_t TorqueShortTermMaximum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueShortTermMaximumMutex);
    m_impl->setTorqueShortTermMaximum(TorqueShortTermMaximum);
}

int32_t VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::getTorqueShortTermMaximum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueShortTermMaximumMutex);
    return m_impl->getTorqueShortTermMaximum();
}
void VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::setTorqueShortTermMinimum(int32_t TorqueShortTermMinimum)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueShortTermMinimumMutex);
    m_impl->setTorqueShortTermMinimum(TorqueShortTermMinimum);
}

int32_t VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::getTorqueShortTermMinimum() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueShortTermMinimumMutex);
    return m_impl->getTorqueShortTermMinimum();
}
void VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::setTorqueMaximumLimit(int32_t TorqueMaximumLimit)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMaximumLimitMutex);
    m_impl->setTorqueMaximumLimit(TorqueMaximumLimit);
}

int32_t VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::getTorqueMaximumLimit() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueMaximumLimitMutex);
    return m_impl->getTorqueMaximumLimit();
}
void VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::setTorqueMinimumLimit(int32_t TorqueMinimumLimit)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMinimumLimitMutex);
    m_impl->setTorqueMinimumLimit(TorqueMinimumLimit);
}

int32_t VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::getTorqueMinimumLimit() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_torqueMinimumLimitMutex);
    return m_impl->getTorqueMinimumLimit();
}
void VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::setRotationalSpeed(int32_t RotationalSpeed)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rotationalSpeedMutex);
    m_impl->setRotationalSpeed(RotationalSpeed);
}

int32_t VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::getRotationalSpeed() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_rotationalSpeedMutex);
    return m_impl->getRotationalSpeed();
}
void VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::setRotationalSpeedTarget(int32_t RotationalSpeedTarget)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rotationalSpeedTargetMutex);
    m_impl->setRotationalSpeedTarget(RotationalSpeedTarget);
}

int32_t VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::getRotationalSpeedTarget() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_rotationalSpeedTargetMutex);
    return m_impl->getRotationalSpeedTarget();
}
void VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::setRotationalSpeedMaximumLimit(int32_t RotationalSpeedMaximumLimit)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rotationalSpeedMaximumLimitMutex);
    m_impl->setRotationalSpeedMaximumLimit(RotationalSpeedMaximumLimit);
}

int32_t VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::getRotationalSpeedMaximumLimit() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_rotationalSpeedMaximumLimitMutex);
    return m_impl->getRotationalSpeedMaximumLimit();
}
void VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::setRotationalSpeedMinimumLimit(int32_t RotationalSpeedMinimumLimit)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rotationalSpeedMinimumLimitMutex);
    m_impl->setRotationalSpeedMinimumLimit(RotationalSpeedMinimumLimit);
}

int32_t VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::getRotationalSpeedMinimumLimit() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_rotationalSpeedMinimumLimitMutex);
    return m_impl->getRotationalSpeedMinimumLimit();
}

IVehicleMotionManagementElectricAxleRow1Publisher& VehicleMotionManagementElectricAxleRow1ThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}