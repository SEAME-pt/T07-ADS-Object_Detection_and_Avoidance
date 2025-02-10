

#include "vehicle/implementation/vehiclemotionmanagementbrakeaxlerow2wheelright.h"
#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow2wheelright.publisher.h"
#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow2wheelright.data.h"

using namespace Cpp::Vehicle;

VehicleMotionManagementBrakeAxleRow2WheelRight::VehicleMotionManagementBrakeAxleRow2WheelRight()
    : m_publisher(std::make_unique<VehicleMotionManagementBrakeAxleRow2WheelRightPublisher>())
{
}
VehicleMotionManagementBrakeAxleRow2WheelRight::~VehicleMotionManagementBrakeAxleRow2WheelRight()
{
}

void VehicleMotionManagementBrakeAxleRow2WheelRight::setTorque(int32_t Torque)
{
    if (m_data.m_Torque != Torque) {
        m_data.m_Torque = Torque;
        m_publisher->publishTorqueChanged(Torque);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelRight::getTorque() const
{
    return m_data.m_Torque;
}

void VehicleMotionManagementBrakeAxleRow2WheelRight::setTorqueArbitrated(int32_t TorqueArbitrated)
{
    if (m_data.m_TorqueArbitrated != TorqueArbitrated) {
        m_data.m_TorqueArbitrated = TorqueArbitrated;
        m_publisher->publishTorqueArbitratedChanged(TorqueArbitrated);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelRight::getTorqueArbitrated() const
{
    return m_data.m_TorqueArbitrated;
}

void VehicleMotionManagementBrakeAxleRow2WheelRight::setTorqueFrictionMaximum(int32_t TorqueFrictionMaximum)
{
    if (m_data.m_TorqueFrictionMaximum != TorqueFrictionMaximum) {
        m_data.m_TorqueFrictionMaximum = TorqueFrictionMaximum;
        m_publisher->publishTorqueFrictionMaximumChanged(TorqueFrictionMaximum);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelRight::getTorqueFrictionMaximum() const
{
    return m_data.m_TorqueFrictionMaximum;
}

void VehicleMotionManagementBrakeAxleRow2WheelRight::setTorqueFrictionMinimum(int32_t TorqueFrictionMinimum)
{
    if (m_data.m_TorqueFrictionMinimum != TorqueFrictionMinimum) {
        m_data.m_TorqueFrictionMinimum = TorqueFrictionMinimum;
        m_publisher->publishTorqueFrictionMinimumChanged(TorqueFrictionMinimum);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelRight::getTorqueFrictionMinimum() const
{
    return m_data.m_TorqueFrictionMinimum;
}

void VehicleMotionManagementBrakeAxleRow2WheelRight::setOmegaUpper(int32_t OmegaUpper)
{
    if (m_data.m_OmegaUpper != OmegaUpper) {
        m_data.m_OmegaUpper = OmegaUpper;
        m_publisher->publishOmegaUpperChanged(OmegaUpper);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelRight::getOmegaUpper() const
{
    return m_data.m_OmegaUpper;
}

void VehicleMotionManagementBrakeAxleRow2WheelRight::setOmegaLower(int32_t OmegaLower)
{
    if (m_data.m_OmegaLower != OmegaLower) {
        m_data.m_OmegaLower = OmegaLower;
        m_publisher->publishOmegaLowerChanged(OmegaLower);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow2WheelRight::getOmegaLower() const
{
    return m_data.m_OmegaLower;
}

IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher& VehicleMotionManagementBrakeAxleRow2WheelRight::_getPublisher() const
{
    return *m_publisher;
}
