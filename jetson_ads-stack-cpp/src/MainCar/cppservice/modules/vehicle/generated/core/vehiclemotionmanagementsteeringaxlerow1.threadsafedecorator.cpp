

#include "vehicle/generated/core/vehiclemotionmanagementsteeringaxlerow1.threadsafedecorator.h"

using namespace Cpp::Vehicle;
VehicleMotionManagementSteeringAxleRow1ThreadSafeDecorator::VehicleMotionManagementSteeringAxleRow1ThreadSafeDecorator(std::shared_ptr<IVehicleMotionManagementSteeringAxleRow1> impl)
    : m_impl(impl)
{
}
void VehicleMotionManagementSteeringAxleRow1ThreadSafeDecorator::setRackPositionOffsetTarget(int32_t RackPositionOffsetTarget)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rackPositionOffsetTargetMutex);
    m_impl->setRackPositionOffsetTarget(RackPositionOffsetTarget);
}

int32_t VehicleMotionManagementSteeringAxleRow1ThreadSafeDecorator::getRackPositionOffsetTarget() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_rackPositionOffsetTargetMutex);
    return m_impl->getRackPositionOffsetTarget();
}
void VehicleMotionManagementSteeringAxleRow1ThreadSafeDecorator::setRackPositionOffsetTargetMode(int32_t RackPositionOffsetTargetMode)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rackPositionOffsetTargetModeMutex);
    m_impl->setRackPositionOffsetTargetMode(RackPositionOffsetTargetMode);
}

int32_t VehicleMotionManagementSteeringAxleRow1ThreadSafeDecorator::getRackPositionOffsetTargetMode() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_rackPositionOffsetTargetModeMutex);
    return m_impl->getRackPositionOffsetTargetMode();
}
void VehicleMotionManagementSteeringAxleRow1ThreadSafeDecorator::setRackPosition(int32_t RackPosition)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rackPositionMutex);
    m_impl->setRackPosition(RackPosition);
}

int32_t VehicleMotionManagementSteeringAxleRow1ThreadSafeDecorator::getRackPosition() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_rackPositionMutex);
    return m_impl->getRackPosition();
}
void VehicleMotionManagementSteeringAxleRow1ThreadSafeDecorator::setRackPositionTarget(int32_t RackPositionTarget)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rackPositionTargetMutex);
    m_impl->setRackPositionTarget(RackPositionTarget);
}

int32_t VehicleMotionManagementSteeringAxleRow1ThreadSafeDecorator::getRackPositionTarget() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_rackPositionTargetMutex);
    return m_impl->getRackPositionTarget();
}
void VehicleMotionManagementSteeringAxleRow1ThreadSafeDecorator::setRackPositionTargetMode(int32_t RackPositionTargetMode)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rackPositionTargetModeMutex);
    m_impl->setRackPositionTargetMode(RackPositionTargetMode);
}

int32_t VehicleMotionManagementSteeringAxleRow1ThreadSafeDecorator::getRackPositionTargetMode() const
{
    std::shared_lock<std::shared_timed_mutex> lock(m_rackPositionTargetModeMutex);
    return m_impl->getRackPositionTargetMode();
}

IVehicleMotionManagementSteeringAxleRow1Publisher& VehicleMotionManagementSteeringAxleRow1ThreadSafeDecorator::_getPublisher() const
{
    return m_impl->_getPublisher();
}