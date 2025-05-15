

#include "vehicle/implementation/vehiclemotionmanagementbrakeaxlerow1wheelright.h"
#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow1wheelright.publisher.h"
#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow1wheelright.data.h"

using namespace Cpp::Vehicle;

VehicleMotionManagementBrakeAxleRow1WheelRight::VehicleMotionManagementBrakeAxleRow1WheelRight()
    : m_publisher(std::make_unique<VehicleMotionManagementBrakeAxleRow1WheelRightPublisher>())
{
}
VehicleMotionManagementBrakeAxleRow1WheelRight::~VehicleMotionManagementBrakeAxleRow1WheelRight()
{
}

void VehicleMotionManagementBrakeAxleRow1WheelRight::setTorque(int32_t Torque)
{
    if (m_data.m_Torque != Torque) {
        m_data.m_Torque = Torque;
        m_publisher->publishTorqueChanged(Torque);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelRight::getTorque() const
{
    return m_data.m_Torque;
}

void VehicleMotionManagementBrakeAxleRow1WheelRight::setTorqueArbitrated(int32_t TorqueArbitrated)
{
    if (m_data.m_TorqueArbitrated != TorqueArbitrated) {
        m_data.m_TorqueArbitrated = TorqueArbitrated;
        m_publisher->publishTorqueArbitratedChanged(TorqueArbitrated);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelRight::getTorqueArbitrated() const
{
    return m_data.m_TorqueArbitrated;
}

void VehicleMotionManagementBrakeAxleRow1WheelRight::setTorqueFrictionMaximum(int32_t TorqueFrictionMaximum)
{
    if (m_data.m_TorqueFrictionMaximum != TorqueFrictionMaximum) {
        m_data.m_TorqueFrictionMaximum = TorqueFrictionMaximum;
        m_publisher->publishTorqueFrictionMaximumChanged(TorqueFrictionMaximum);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelRight::getTorqueFrictionMaximum() const
{
    return m_data.m_TorqueFrictionMaximum;
}

void VehicleMotionManagementBrakeAxleRow1WheelRight::setTorqueFrictionMinimum(int32_t TorqueFrictionMinimum)
{
    if (m_data.m_TorqueFrictionMinimum != TorqueFrictionMinimum) {
        m_data.m_TorqueFrictionMinimum = TorqueFrictionMinimum;
        m_publisher->publishTorqueFrictionMinimumChanged(TorqueFrictionMinimum);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelRight::getTorqueFrictionMinimum() const
{
    return m_data.m_TorqueFrictionMinimum;
}

void VehicleMotionManagementBrakeAxleRow1WheelRight::setOmegaUpper(int32_t OmegaUpper)
{
    if (m_data.m_OmegaUpper != OmegaUpper) {
        m_data.m_OmegaUpper = OmegaUpper;
        m_publisher->publishOmegaUpperChanged(OmegaUpper);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelRight::getOmegaUpper() const
{
    return m_data.m_OmegaUpper;
}

void VehicleMotionManagementBrakeAxleRow1WheelRight::setOmegaLower(int32_t OmegaLower)
{
    if (m_data.m_OmegaLower != OmegaLower) {
        m_data.m_OmegaLower = OmegaLower;
        m_publisher->publishOmegaLowerChanged(OmegaLower);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow1WheelRight::getOmegaLower() const
{
    return m_data.m_OmegaLower;
}

IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher& VehicleMotionManagementBrakeAxleRow1WheelRight::_getPublisher() const
{
    return *m_publisher;
}
