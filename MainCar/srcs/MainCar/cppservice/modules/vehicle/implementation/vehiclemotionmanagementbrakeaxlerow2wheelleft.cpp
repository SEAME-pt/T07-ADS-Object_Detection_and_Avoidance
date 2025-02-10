

#include "vehicle/implementation/vehiclemotionmanagementbrakeaxlerow2wheelleft.h"
#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow2wheelleft.publisher.h"
#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow2wheelleft.data.h"

using namespace Cpp::Vehicle;

VehicleMotionManagementBrakeAxleRow2WheelLeft::VehicleMotionManagementBrakeAxleRow2WheelLeft()
    : m_publisher(std::make_unique<VehicleMotionManagementBrakeAxleRow2WheelLeftPublisher>())
{
}
VehicleMotionManagementBrakeAxleRow2WheelLeft::~VehicleMotionManagementBrakeAxleRow2WheelLeft()
{
}

void VehicleMotionManagementBrakeAxleRow2WheelLeft::setTorque(int32_t Torque)
{
    if (m_data.m_Torque != Torque) {
        m_data.m_Torque = Torque;
        m_publisher->publishTorqueChanged(Torque);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelLeft::getTorque() const
{
    return m_data.m_Torque;
}

void VehicleMotionManagementBrakeAxleRow2WheelLeft::setTorqueArbitrated(int32_t TorqueArbitrated)
{
    if (m_data.m_TorqueArbitrated != TorqueArbitrated) {
        m_data.m_TorqueArbitrated = TorqueArbitrated;
        m_publisher->publishTorqueArbitratedChanged(TorqueArbitrated);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelLeft::getTorqueArbitrated() const
{
    return m_data.m_TorqueArbitrated;
}

void VehicleMotionManagementBrakeAxleRow2WheelLeft::setTorqueFrictionMaximum(int32_t TorqueFrictionMaximum)
{
    if (m_data.m_TorqueFrictionMaximum != TorqueFrictionMaximum) {
        m_data.m_TorqueFrictionMaximum = TorqueFrictionMaximum;
        m_publisher->publishTorqueFrictionMaximumChanged(TorqueFrictionMaximum);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelLeft::getTorqueFrictionMaximum() const
{
    return m_data.m_TorqueFrictionMaximum;
}

void VehicleMotionManagementBrakeAxleRow2WheelLeft::setTorqueFrictionMinimum(int32_t TorqueFrictionMinimum)
{
    if (m_data.m_TorqueFrictionMinimum != TorqueFrictionMinimum) {
        m_data.m_TorqueFrictionMinimum = TorqueFrictionMinimum;
        m_publisher->publishTorqueFrictionMinimumChanged(TorqueFrictionMinimum);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelLeft::getTorqueFrictionMinimum() const
{
    return m_data.m_TorqueFrictionMinimum;
}

void VehicleMotionManagementBrakeAxleRow2WheelLeft::setOmegaUpper(int32_t OmegaUpper)
{
    if (m_data.m_OmegaUpper != OmegaUpper) {
        m_data.m_OmegaUpper = OmegaUpper;
        m_publisher->publishOmegaUpperChanged(OmegaUpper);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelLeft::getOmegaUpper() const
{
    return m_data.m_OmegaUpper;
}

void VehicleMotionManagementBrakeAxleRow2WheelLeft::setOmegaLower(int32_t OmegaLower)
{
    if (m_data.m_OmegaLower != OmegaLower) {
        m_data.m_OmegaLower = OmegaLower;
        m_publisher->publishOmegaLowerChanged(OmegaLower);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelLeft::getOmegaLower() const
{
    return m_data.m_OmegaLower;
}

IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher& VehicleMotionManagementBrakeAxleRow2WheelLeft::_getPublisher() const
{
    return *m_publisher;
}
