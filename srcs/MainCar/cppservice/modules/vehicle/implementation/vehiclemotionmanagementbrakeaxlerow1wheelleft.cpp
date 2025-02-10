

#include "vehicle/implementation/vehiclemotionmanagementbrakeaxlerow1wheelleft.h"
#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow1wheelleft.publisher.h"
#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow1wheelleft.data.h"

using namespace Cpp::Vehicle;

VehicleMotionManagementBrakeAxleRow1WheelLeft::VehicleMotionManagementBrakeAxleRow1WheelLeft()
    : m_publisher(std::make_unique<VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher>())
{
}
VehicleMotionManagementBrakeAxleRow1WheelLeft::~VehicleMotionManagementBrakeAxleRow1WheelLeft()
{
}

void VehicleMotionManagementBrakeAxleRow1WheelLeft::setTorque(int32_t Torque)
{
    if (m_data.m_Torque != Torque) {
        m_data.m_Torque = Torque;
        m_publisher->publishTorqueChanged(Torque);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelLeft::getTorque() const
{
    return m_data.m_Torque;
}

void VehicleMotionManagementBrakeAxleRow1WheelLeft::setTorqueArbitrated(int32_t TorqueArbitrated)
{
    if (m_data.m_TorqueArbitrated != TorqueArbitrated) {
        m_data.m_TorqueArbitrated = TorqueArbitrated;
        m_publisher->publishTorqueArbitratedChanged(TorqueArbitrated);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelLeft::getTorqueArbitrated() const
{
    return m_data.m_TorqueArbitrated;
}

void VehicleMotionManagementBrakeAxleRow1WheelLeft::setTorqueFrictionMaximum(int32_t TorqueFrictionMaximum)
{
    if (m_data.m_TorqueFrictionMaximum != TorqueFrictionMaximum) {
        m_data.m_TorqueFrictionMaximum = TorqueFrictionMaximum;
        m_publisher->publishTorqueFrictionMaximumChanged(TorqueFrictionMaximum);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelLeft::getTorqueFrictionMaximum() const
{
    return m_data.m_TorqueFrictionMaximum;
}

void VehicleMotionManagementBrakeAxleRow1WheelLeft::setTorqueFrictionMinimum(int32_t TorqueFrictionMinimum)
{
    if (m_data.m_TorqueFrictionMinimum != TorqueFrictionMinimum) {
        m_data.m_TorqueFrictionMinimum = TorqueFrictionMinimum;
        m_publisher->publishTorqueFrictionMinimumChanged(TorqueFrictionMinimum);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelLeft::getTorqueFrictionMinimum() const
{
    return m_data.m_TorqueFrictionMinimum;
}

void VehicleMotionManagementBrakeAxleRow1WheelLeft::setOmegaUpper(int32_t OmegaUpper)
{
    if (m_data.m_OmegaUpper != OmegaUpper) {
        m_data.m_OmegaUpper = OmegaUpper;
        m_publisher->publishOmegaUpperChanged(OmegaUpper);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelLeft::getOmegaUpper() const
{
    return m_data.m_OmegaUpper;
}

void VehicleMotionManagementBrakeAxleRow1WheelLeft::setOmegaLower(int32_t OmegaLower)
{
    if (m_data.m_OmegaLower != OmegaLower) {
        m_data.m_OmegaLower = OmegaLower;
        m_publisher->publishOmegaLowerChanged(OmegaLower);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelLeft::getOmegaLower() const
{
    return m_data.m_OmegaLower;
}

IVehicleMotionManagementBrakeAxleRow1WheelLeftPublisher& VehicleMotionManagementBrakeAxleRow1WheelLeft::_getPublisher() const
{
    return *m_publisher;
}
