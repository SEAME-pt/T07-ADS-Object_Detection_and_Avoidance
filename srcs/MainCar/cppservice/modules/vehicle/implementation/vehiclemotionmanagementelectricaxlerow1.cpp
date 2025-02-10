

#include "vehicle/implementation/vehiclemotionmanagementelectricaxlerow1.h"
#include "vehicle/generated/core/vehiclemotionmanagementelectricaxlerow1.publisher.h"
#include "vehicle/generated/core/vehiclemotionmanagementelectricaxlerow1.data.h"

using namespace Cpp::Vehicle;

VehicleMotionManagementElectricAxleRow1::VehicleMotionManagementElectricAxleRow1()
    : m_publisher(std::make_unique<VehicleMotionManagementElectricAxleRow1Publisher>())
{
}
VehicleMotionManagementElectricAxleRow1::~VehicleMotionManagementElectricAxleRow1()
{
}

void VehicleMotionManagementElectricAxleRow1::setTorque(int32_t Torque)
{
    if (m_data.m_Torque != Torque) {
        m_data.m_Torque = Torque;
        m_publisher->publishTorqueChanged(Torque);
    }
}

int32_t VehicleMotionManagementElectricAxleRow1::getTorque() const
{
    return m_data.m_Torque;
}

void VehicleMotionManagementElectricAxleRow1::setTorqueTarget(int32_t TorqueTarget)
{
    if (m_data.m_TorqueTarget != TorqueTarget) {
        m_data.m_TorqueTarget = TorqueTarget;
        m_publisher->publishTorqueTargetChanged(TorqueTarget);
    }
}

int32_t VehicleMotionManagementElectricAxleRow1::getTorqueTarget() const
{
    return m_data.m_TorqueTarget;
}

void VehicleMotionManagementElectricAxleRow1::setTorqueMaximum(int32_t TorqueMaximum)
{
    if (m_data.m_TorqueMaximum != TorqueMaximum) {
        m_data.m_TorqueMaximum = TorqueMaximum;
        m_publisher->publishTorqueMaximumChanged(TorqueMaximum);
    }
}

int32_t VehicleMotionManagementElectricAxleRow1::getTorqueMaximum() const
{
    return m_data.m_TorqueMaximum;
}

void VehicleMotionManagementElectricAxleRow1::setTorqueMinimum(int32_t TorqueMinimum)
{
    if (m_data.m_TorqueMinimum != TorqueMinimum) {
        m_data.m_TorqueMinimum = TorqueMinimum;
        m_publisher->publishTorqueMinimumChanged(TorqueMinimum);
    }
}

int32_t VehicleMotionManagementElectricAxleRow1::getTorqueMinimum() const
{
    return m_data.m_TorqueMinimum;
}

void VehicleMotionManagementElectricAxleRow1::setTorqueLongTermMaximum(int32_t TorqueLongTermMaximum)
{
    if (m_data.m_TorqueLongTermMaximum != TorqueLongTermMaximum) {
        m_data.m_TorqueLongTermMaximum = TorqueLongTermMaximum;
        m_publisher->publishTorqueLongTermMaximumChanged(TorqueLongTermMaximum);
    }
}

int32_t VehicleMotionManagementElectricAxleRow1::getTorqueLongTermMaximum() const
{
    return m_data.m_TorqueLongTermMaximum;
}

void VehicleMotionManagementElectricAxleRow1::setTorqueLongTermMinimum(int32_t TorqueLongTermMinimum)
{
    if (m_data.m_TorqueLongTermMinimum != TorqueLongTermMinimum) {
        m_data.m_TorqueLongTermMinimum = TorqueLongTermMinimum;
        m_publisher->publishTorqueLongTermMinimumChanged(TorqueLongTermMinimum);
    }
}

int32_t VehicleMotionManagementElectricAxleRow1::getTorqueLongTermMinimum() const
{
    return m_data.m_TorqueLongTermMinimum;
}

void VehicleMotionManagementElectricAxleRow1::setTorqueShortTermMaximum(int32_t TorqueShortTermMaximum)
{
    if (m_data.m_TorqueShortTermMaximum != TorqueShortTermMaximum) {
        m_data.m_TorqueShortTermMaximum = TorqueShortTermMaximum;
        m_publisher->publishTorqueShortTermMaximumChanged(TorqueShortTermMaximum);
    }
}

int32_t VehicleMotionManagementElectricAxleRow1::getTorqueShortTermMaximum() const
{
    return m_data.m_TorqueShortTermMaximum;
}

void VehicleMotionManagementElectricAxleRow1::setTorqueShortTermMinimum(int32_t TorqueShortTermMinimum)
{
    if (m_data.m_TorqueShortTermMinimum != TorqueShortTermMinimum) {
        m_data.m_TorqueShortTermMinimum = TorqueShortTermMinimum;
        m_publisher->publishTorqueShortTermMinimumChanged(TorqueShortTermMinimum);
    }
}

int32_t VehicleMotionManagementElectricAxleRow1::getTorqueShortTermMinimum() const
{
    return m_data.m_TorqueShortTermMinimum;
}

void VehicleMotionManagementElectricAxleRow1::setTorqueMaximumLimit(int32_t TorqueMaximumLimit)
{
    if (m_data.m_TorqueMaximumLimit != TorqueMaximumLimit) {
        m_data.m_TorqueMaximumLimit = TorqueMaximumLimit;
        m_publisher->publishTorqueMaximumLimitChanged(TorqueMaximumLimit);
    }
}

int32_t VehicleMotionManagementElectricAxleRow1::getTorqueMaximumLimit() const
{
    return m_data.m_TorqueMaximumLimit;
}

void VehicleMotionManagementElectricAxleRow1::setTorqueMinimumLimit(int32_t TorqueMinimumLimit)
{
    if (m_data.m_TorqueMinimumLimit != TorqueMinimumLimit) {
        m_data.m_TorqueMinimumLimit = TorqueMinimumLimit;
        m_publisher->publishTorqueMinimumLimitChanged(TorqueMinimumLimit);
    }
}

int32_t VehicleMotionManagementElectricAxleRow1::getTorqueMinimumLimit() const
{
    return m_data.m_TorqueMinimumLimit;
}

void VehicleMotionManagementElectricAxleRow1::setRotationalSpeed(int32_t RotationalSpeed)
{
    if (m_data.m_RotationalSpeed != RotationalSpeed) {
        m_data.m_RotationalSpeed = RotationalSpeed;
        m_publisher->publishRotationalSpeedChanged(RotationalSpeed);
    }
}

int32_t VehicleMotionManagementElectricAxleRow1::getRotationalSpeed() const
{
    return m_data.m_RotationalSpeed;
}

void VehicleMotionManagementElectricAxleRow1::setRotationalSpeedTarget(int32_t RotationalSpeedTarget)
{
    if (m_data.m_RotationalSpeedTarget != RotationalSpeedTarget) {
        m_data.m_RotationalSpeedTarget = RotationalSpeedTarget;
        m_publisher->publishRotationalSpeedTargetChanged(RotationalSpeedTarget);
    }
}

int32_t VehicleMotionManagementElectricAxleRow1::getRotationalSpeedTarget() const
{
    return m_data.m_RotationalSpeedTarget;
}

void VehicleMotionManagementElectricAxleRow1::setRotationalSpeedMaximumLimit(int32_t RotationalSpeedMaximumLimit)
{
    if (m_data.m_RotationalSpeedMaximumLimit != RotationalSpeedMaximumLimit) {
        m_data.m_RotationalSpeedMaximumLimit = RotationalSpeedMaximumLimit;
        m_publisher->publishRotationalSpeedMaximumLimitChanged(RotationalSpeedMaximumLimit);
    }
}

int32_t VehicleMotionManagementElectricAxleRow1::getRotationalSpeedMaximumLimit() const
{
    return m_data.m_RotationalSpeedMaximumLimit;
}

void VehicleMotionManagementElectricAxleRow1::setRotationalSpeedMinimumLimit(int32_t RotationalSpeedMinimumLimit)
{
    if (m_data.m_RotationalSpeedMinimumLimit != RotationalSpeedMinimumLimit) {
        m_data.m_RotationalSpeedMinimumLimit = RotationalSpeedMinimumLimit;
        m_publisher->publishRotationalSpeedMinimumLimitChanged(RotationalSpeedMinimumLimit);
    }
}

int32_t VehicleMotionManagementElectricAxleRow1::getRotationalSpeedMinimumLimit() const
{
    return m_data.m_RotationalSpeedMinimumLimit;
}

IVehicleMotionManagementElectricAxleRow1Publisher& VehicleMotionManagementElectricAxleRow1::_getPublisher() const
{
    return *m_publisher;
}
