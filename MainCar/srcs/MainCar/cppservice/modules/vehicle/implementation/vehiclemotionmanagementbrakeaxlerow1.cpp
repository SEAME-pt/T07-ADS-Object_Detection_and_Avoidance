

#include "vehicle/implementation/vehiclemotionmanagementbrakeaxlerow1.h"
#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow1.publisher.h"
#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow1.data.h"

using namespace Cpp::Vehicle;

VehicleMotionManagementBrakeAxleRow1::VehicleMotionManagementBrakeAxleRow1()
    : m_publisher(std::make_unique<VehicleMotionManagementBrakeAxleRow1Publisher>())
{
}
VehicleMotionManagementBrakeAxleRow1::~VehicleMotionManagementBrakeAxleRow1()
{
}

void VehicleMotionManagementBrakeAxleRow1::setTorqueElectricMinimum(int32_t TorqueElectricMinimum)
{
    if (m_data.m_TorqueElectricMinimum != TorqueElectricMinimum) {
        m_data.m_TorqueElectricMinimum = TorqueElectricMinimum;
        m_publisher->publishTorqueElectricMinimumChanged(TorqueElectricMinimum);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow1::getTorqueElectricMinimum() const
{
    return m_data.m_TorqueElectricMinimum;
}

void VehicleMotionManagementBrakeAxleRow1::setTorqueFrictionDifferenceMaximum(int32_t TorqueFrictionDifferenceMaximum)
{
    if (m_data.m_TorqueFrictionDifferenceMaximum != TorqueFrictionDifferenceMaximum) {
        m_data.m_TorqueFrictionDifferenceMaximum = TorqueFrictionDifferenceMaximum;
        m_publisher->publishTorqueFrictionDifferenceMaximumChanged(TorqueFrictionDifferenceMaximum);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow1::getTorqueFrictionDifferenceMaximum() const
{
    return m_data.m_TorqueFrictionDifferenceMaximum;
}

void VehicleMotionManagementBrakeAxleRow1::setTorqueDistributionFrictionRightMaximum(int32_t TorqueDistributionFrictionRightMaximum)
{
    if (m_data.m_TorqueDistributionFrictionRightMaximum != TorqueDistributionFrictionRightMaximum) {
        m_data.m_TorqueDistributionFrictionRightMaximum = TorqueDistributionFrictionRightMaximum;
        m_publisher->publishTorqueDistributionFrictionRightMaximumChanged(TorqueDistributionFrictionRightMaximum);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow1::getTorqueDistributionFrictionRightMaximum() const
{
    return m_data.m_TorqueDistributionFrictionRightMaximum;
}

void VehicleMotionManagementBrakeAxleRow1::setTorqueDistributionFrictionRightMinimum(int32_t TorqueDistributionFrictionRightMinimum)
{
    if (m_data.m_TorqueDistributionFrictionRightMinimum != TorqueDistributionFrictionRightMinimum) {
        m_data.m_TorqueDistributionFrictionRightMinimum = TorqueDistributionFrictionRightMinimum;
        m_publisher->publishTorqueDistributionFrictionRightMinimumChanged(TorqueDistributionFrictionRightMinimum);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow1::getTorqueDistributionFrictionRightMinimum() const
{
    return m_data.m_TorqueDistributionFrictionRightMinimum;
}

IVehicleMotionManagementBrakeAxleRow1Publisher& VehicleMotionManagementBrakeAxleRow1::_getPublisher() const
{
    return *m_publisher;
}
