

#include "vehicle/implementation/vehiclemotionmanagementelectricaxlerow2.h"
#include "vehicle/generated/core/vehiclemotionmanagementelectricaxlerow2.publisher.h"
#include "vehicle/generated/core/vehiclemotionmanagementelectricaxlerow2.data.h"

using namespace Cpp::Vehicle;

VehicleMotionManagementElectricAxleRow2::VehicleMotionManagementElectricAxleRow2()
    : m_publisher(std::make_unique<VehicleMotionManagementElectricAxleRow2Publisher>())
{
}
VehicleMotionManagementElectricAxleRow2::~VehicleMotionManagementElectricAxleRow2()
{
}

void VehicleMotionManagementElectricAxleRow2::setTorque(int32_t Torque)
{
    if (m_data.m_Torque != Torque) {
        m_data.m_Torque = Torque;
        m_publisher->publishTorqueChanged(Torque);
    }
}

int32_t VehicleMotionManagementElectricAxleRow2::getTorque() const
{
    return m_data.m_Torque;
}

void VehicleMotionManagementElectricAxleRow2::setTorqueTarget(int32_t TorqueTarget)
{
    if (m_data.m_TorqueTarget != TorqueTarget) {
        m_data.m_TorqueTarget = TorqueTarget;
        m_publisher->publishTorqueTargetChanged(TorqueTarget);
    }
}

int32_t VehicleMotionManagementElectricAxleRow2::getTorqueTarget() const
{
    return m_data.m_TorqueTarget;
}

void VehicleMotionManagementElectricAxleRow2::setTorqueMaximum(int32_t TorqueMaximum)
{
    if (m_data.m_TorqueMaximum != TorqueMaximum) {
        m_data.m_TorqueMaximum = TorqueMaximum;
        m_publisher->publishTorqueMaximumChanged(TorqueMaximum);
    }
}

int32_t VehicleMotionManagementElectricAxleRow2::getTorqueMaximum() const
{
    return m_data.m_TorqueMaximum;
}

void VehicleMotionManagementElectricAxleRow2::setTorqueMinimum(int32_t TorqueMinimum)
{
    if (m_data.m_TorqueMinimum != TorqueMinimum) {
        m_data.m_TorqueMinimum = TorqueMinimum;
        m_publisher->publishTorqueMinimumChanged(TorqueMinimum);
    }
}

int32_t VehicleMotionManagementElectricAxleRow2::getTorqueMinimum() const
{
    return m_data.m_TorqueMinimum;
}

void VehicleMotionManagementElectricAxleRow2::setTorqueLongTermMaximum(int32_t TorqueLongTermMaximum)
{
    if (m_data.m_TorqueLongTermMaximum != TorqueLongTermMaximum) {
        m_data.m_TorqueLongTermMaximum = TorqueLongTermMaximum;
        m_publisher->publishTorqueLongTermMaximumChanged(TorqueLongTermMaximum);
    }
}

int32_t VehicleMotionManagementElectricAxleRow2::getTorqueLongTermMaximum() const
{
    return m_data.m_TorqueLongTermMaximum;
}

void VehicleMotionManagementElectricAxleRow2::setTorqueLongTermMinimum(int32_t TorqueLongTermMinimum)
{
    if (m_data.m_TorqueLongTermMinimum != TorqueLongTermMinimum) {
        m_data.m_TorqueLongTermMinimum = TorqueLongTermMinimum;
        m_publisher->publishTorqueLongTermMinimumChanged(TorqueLongTermMinimum);
    }
}

int32_t VehicleMotionManagementElectricAxleRow2::getTorqueLongTermMinimum() const
{
    return m_data.m_TorqueLongTermMinimum;
}

void VehicleMotionManagementElectricAxleRow2::setTorqueShortTermMaximum(int32_t TorqueShortTermMaximum)
{
    if (m_data.m_TorqueShortTermMaximum != TorqueShortTermMaximum) {
        m_data.m_TorqueShortTermMaximum = TorqueShortTermMaximum;
        m_publisher->publishTorqueShortTermMaximumChanged(TorqueShortTermMaximum);
    }
}

int32_t VehicleMotionManagementElectricAxleRow2::getTorqueShortTermMaximum() const
{
    return m_data.m_TorqueShortTermMaximum;
}

void VehicleMotionManagementElectricAxleRow2::setTorqueShortTermMinimum(int32_t TorqueShortTermMinimum)
{
    if (m_data.m_TorqueShortTermMinimum != TorqueShortTermMinimum) {
        m_data.m_TorqueShortTermMinimum = TorqueShortTermMinimum;
        m_publisher->publishTorqueShortTermMinimumChanged(TorqueShortTermMinimum);
    }
}

int32_t VehicleMotionManagementElectricAxleRow2::getTorqueShortTermMinimum() const
{
    return m_data.m_TorqueShortTermMinimum;
}

void VehicleMotionManagementElectricAxleRow2::setTorqueMaximumLimit(int32_t TorqueMaximumLimit)
{
    if (m_data.m_TorqueMaximumLimit != TorqueMaximumLimit) {
        m_data.m_TorqueMaximumLimit = TorqueMaximumLimit;
        m_publisher->publishTorqueMaximumLimitChanged(TorqueMaximumLimit);
    }
}

int32_t VehicleMotionManagementElectricAxleRow2::getTorqueMaximumLimit() const
{
    return m_data.m_TorqueMaximumLimit;
}

void VehicleMotionManagementElectricAxleRow2::setTorqueMinimumLimit(int32_t TorqueMinimumLimit)
{
    if (m_data.m_TorqueMinimumLimit != TorqueMinimumLimit) {
        m_data.m_TorqueMinimumLimit = TorqueMinimumLimit;
        m_publisher->publishTorqueMinimumLimitChanged(TorqueMinimumLimit);
    }
}

int32_t VehicleMotionManagementElectricAxleRow2::getTorqueMinimumLimit() const
{
    return m_data.m_TorqueMinimumLimit;
}

void VehicleMotionManagementElectricAxleRow2::setRotationalSpeed(int32_t RotationalSpeed)
{
    if (m_data.m_RotationalSpeed != RotationalSpeed) {
        m_data.m_RotationalSpeed = RotationalSpeed;
        m_publisher->publishRotationalSpeedChanged(RotationalSpeed);
    }
}

int32_t VehicleMotionManagementElectricAxleRow2::getRotationalSpeed() const
{
    return m_data.m_RotationalSpeed;
}

void VehicleMotionManagementElectricAxleRow2::setRotationalSpeedTarget(int32_t RotationalSpeedTarget)
{
    if (m_data.m_RotationalSpeedTarget != RotationalSpeedTarget) {
        m_data.m_RotationalSpeedTarget = RotationalSpeedTarget;
        m_publisher->publishRotationalSpeedTargetChanged(RotationalSpeedTarget);
    }
}

int32_t VehicleMotionManagementElectricAxleRow2::getRotationalSpeedTarget() const
{
    return m_data.m_RotationalSpeedTarget;
}

void VehicleMotionManagementElectricAxleRow2::setRotationalSpeedMaximumLimit(int32_t RotationalSpeedMaximumLimit)
{
    if (m_data.m_RotationalSpeedMaximumLimit != RotationalSpeedMaximumLimit) {
        m_data.m_RotationalSpeedMaximumLimit = RotationalSpeedMaximumLimit;
        m_publisher->publishRotationalSpeedMaximumLimitChanged(RotationalSpeedMaximumLimit);
    }
}

int32_t VehicleMotionManagementElectricAxleRow2::getRotationalSpeedMaximumLimit() const
{
    return m_data.m_RotationalSpeedMaximumLimit;
}

void VehicleMotionManagementElectricAxleRow2::setRotationalSpeedMinimumLimit(int32_t RotationalSpeedMinimumLimit)
{
    if (m_data.m_RotationalSpeedMinimumLimit != RotationalSpeedMinimumLimit) {
        m_data.m_RotationalSpeedMinimumLimit = RotationalSpeedMinimumLimit;
        m_publisher->publishRotationalSpeedMinimumLimitChanged(RotationalSpeedMinimumLimit);
    }
}

int32_t VehicleMotionManagementElectricAxleRow2::getRotationalSpeedMinimumLimit() const
{
    return m_data.m_RotationalSpeedMinimumLimit;
}

IVehicleMotionManagementElectricAxleRow2Publisher& VehicleMotionManagementElectricAxleRow2::_getPublisher() const
{
    return *m_publisher;
}
