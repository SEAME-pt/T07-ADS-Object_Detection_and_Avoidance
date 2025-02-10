

#include "vehicle/implementation/vehiclemotionmanagementsteeringaxlerow1.h"
#include "vehicle/generated/core/vehiclemotionmanagementsteeringaxlerow1.publisher.h"
#include "vehicle/generated/core/vehiclemotionmanagementsteeringaxlerow1.data.h"

using namespace Cpp::Vehicle;

VehicleMotionManagementSteeringAxleRow1::VehicleMotionManagementSteeringAxleRow1()
    : m_publisher(std::make_unique<VehicleMotionManagementSteeringAxleRow1Publisher>())
{
}
VehicleMotionManagementSteeringAxleRow1::~VehicleMotionManagementSteeringAxleRow1()
{
}

void VehicleMotionManagementSteeringAxleRow1::setRackPositionOffsetTarget(int32_t RackPositionOffsetTarget)
{
    if (m_data.m_RackPositionOffsetTarget != RackPositionOffsetTarget) {
        m_data.m_RackPositionOffsetTarget = RackPositionOffsetTarget;
        m_publisher->publishRackPositionOffsetTargetChanged(RackPositionOffsetTarget);
    }
}

int32_t VehicleMotionManagementSteeringAxleRow1::getRackPositionOffsetTarget() const
{
    return m_data.m_RackPositionOffsetTarget;
}

void VehicleMotionManagementSteeringAxleRow1::setRackPositionOffsetTargetMode(int32_t RackPositionOffsetTargetMode)
{
    if (m_data.m_RackPositionOffsetTargetMode != RackPositionOffsetTargetMode) {
        m_data.m_RackPositionOffsetTargetMode = RackPositionOffsetTargetMode;
        m_publisher->publishRackPositionOffsetTargetModeChanged(RackPositionOffsetTargetMode);
    }
}

int32_t VehicleMotionManagementSteeringAxleRow1::getRackPositionOffsetTargetMode() const
{
    return m_data.m_RackPositionOffsetTargetMode;
}

void VehicleMotionManagementSteeringAxleRow1::setRackPosition(int32_t RackPosition)
{
    if (m_data.m_RackPosition != RackPosition) {
        m_data.m_RackPosition = RackPosition;
        m_publisher->publishRackPositionChanged(RackPosition);
    }
}

int32_t VehicleMotionManagementSteeringAxleRow1::getRackPosition() const
{
    return m_data.m_RackPosition;
}

void VehicleMotionManagementSteeringAxleRow1::setRackPositionTarget(int32_t RackPositionTarget)
{
    if (m_data.m_RackPositionTarget != RackPositionTarget) {
        m_data.m_RackPositionTarget = RackPositionTarget;
        m_publisher->publishRackPositionTargetChanged(RackPositionTarget);
    }
}

int32_t VehicleMotionManagementSteeringAxleRow1::getRackPositionTarget() const
{
    return m_data.m_RackPositionTarget;
}

void VehicleMotionManagementSteeringAxleRow1::setRackPositionTargetMode(int32_t RackPositionTargetMode)
{
    if (m_data.m_RackPositionTargetMode != RackPositionTargetMode) {
        m_data.m_RackPositionTargetMode = RackPositionTargetMode;
        m_publisher->publishRackPositionTargetModeChanged(RackPositionTargetMode);
    }
}

int32_t VehicleMotionManagementSteeringAxleRow1::getRackPositionTargetMode() const
{
    return m_data.m_RackPositionTargetMode;
}

IVehicleMotionManagementSteeringAxleRow1Publisher& VehicleMotionManagementSteeringAxleRow1::_getPublisher() const
{
    return *m_publisher;
}
