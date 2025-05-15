

#include "vehicle/implementation/vehiclemotionmanagementsteeringsteeringwheel.h"
#include "vehicle/generated/core/vehiclemotionmanagementsteeringsteeringwheel.publisher.h"
#include "vehicle/generated/core/vehiclemotionmanagementsteeringsteeringwheel.data.h"

using namespace Cpp::Vehicle;

VehicleMotionManagementSteeringSteeringWheel::VehicleMotionManagementSteeringSteeringWheel()
    : m_publisher(std::make_unique<VehicleMotionManagementSteeringSteeringWheelPublisher>())
{
}
VehicleMotionManagementSteeringSteeringWheel::~VehicleMotionManagementSteeringSteeringWheel()
{
}

void VehicleMotionManagementSteeringSteeringWheel::setAngle(int32_t Angle)
{
    if (m_data.m_Angle != Angle) {
        m_data.m_Angle = Angle;
        m_publisher->publishAngleChanged(Angle);
    }
}

int32_t VehicleMotionManagementSteeringSteeringWheel::getAngle() const
{
    return m_data.m_Angle;
}

void VehicleMotionManagementSteeringSteeringWheel::setAngleTarget(int32_t AngleTarget)
{
    if (m_data.m_AngleTarget != AngleTarget) {
        m_data.m_AngleTarget = AngleTarget;
        m_publisher->publishAngleTargetChanged(AngleTarget);
    }
}

int32_t VehicleMotionManagementSteeringSteeringWheel::getAngleTarget() const
{
    return m_data.m_AngleTarget;
}

void VehicleMotionManagementSteeringSteeringWheel::setAngleTargetMode(int32_t AngleTargetMode)
{
    if (m_data.m_AngleTargetMode != AngleTargetMode) {
        m_data.m_AngleTargetMode = AngleTargetMode;
        m_publisher->publishAngleTargetModeChanged(AngleTargetMode);
    }
}

int32_t VehicleMotionManagementSteeringSteeringWheel::getAngleTargetMode() const
{
    return m_data.m_AngleTargetMode;
}

void VehicleMotionManagementSteeringSteeringWheel::setTorqueOffsetTarget(int32_t TorqueOffsetTarget)
{
    if (m_data.m_TorqueOffsetTarget != TorqueOffsetTarget) {
        m_data.m_TorqueOffsetTarget = TorqueOffsetTarget;
        m_publisher->publishTorqueOffsetTargetChanged(TorqueOffsetTarget);
    }
}

int32_t VehicleMotionManagementSteeringSteeringWheel::getTorqueOffsetTarget() const
{
    return m_data.m_TorqueOffsetTarget;
}

void VehicleMotionManagementSteeringSteeringWheel::setTorqueOffsetTargetMode(int32_t TorqueOffsetTargetMode)
{
    if (m_data.m_TorqueOffsetTargetMode != TorqueOffsetTargetMode) {
        m_data.m_TorqueOffsetTargetMode = TorqueOffsetTargetMode;
        m_publisher->publishTorqueOffsetTargetModeChanged(TorqueOffsetTargetMode);
    }
}

int32_t VehicleMotionManagementSteeringSteeringWheel::getTorqueOffsetTargetMode() const
{
    return m_data.m_TorqueOffsetTargetMode;
}

void VehicleMotionManagementSteeringSteeringWheel::setTorqueTarget(int32_t TorqueTarget)
{
    if (m_data.m_TorqueTarget != TorqueTarget) {
        m_data.m_TorqueTarget = TorqueTarget;
        m_publisher->publishTorqueTargetChanged(TorqueTarget);
    }
}

int32_t VehicleMotionManagementSteeringSteeringWheel::getTorqueTarget() const
{
    return m_data.m_TorqueTarget;
}

void VehicleMotionManagementSteeringSteeringWheel::setTorqueTargetMode(int32_t TorqueTargetMode)
{
    if (m_data.m_TorqueTargetMode != TorqueTargetMode) {
        m_data.m_TorqueTargetMode = TorqueTargetMode;
        m_publisher->publishTorqueTargetModeChanged(TorqueTargetMode);
    }
}

int32_t VehicleMotionManagementSteeringSteeringWheel::getTorqueTargetMode() const
{
    return m_data.m_TorqueTargetMode;
}

void VehicleMotionManagementSteeringSteeringWheel::setTorque(int32_t Torque)
{
    if (m_data.m_Torque != Torque) {
        m_data.m_Torque = Torque;
        m_publisher->publishTorqueChanged(Torque);
    }
}

int32_t VehicleMotionManagementSteeringSteeringWheel::getTorque() const
{
    return m_data.m_Torque;
}

IVehicleMotionManagementSteeringSteeringWheelPublisher& VehicleMotionManagementSteeringSteeringWheel::_getPublisher() const
{
    return *m_publisher;
}
