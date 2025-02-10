

#include "vehicle/implementation/vehiclemotionmanagementbrakeaxlerow2.h"
#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow2.publisher.h"
#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow2.data.h"

using namespace Cpp::Vehicle;

VehicleMotionManagementBrakeAxleRow2::VehicleMotionManagementBrakeAxleRow2()
    : m_publisher(std::make_unique<VehicleMotionManagementBrakeAxleRow2Publisher>())
{
}
VehicleMotionManagementBrakeAxleRow2::~VehicleMotionManagementBrakeAxleRow2()
{
}

void VehicleMotionManagementBrakeAxleRow2::setTorqueElectricMinimum(int32_t TorqueElectricMinimum)
{
    if (m_data.m_TorqueElectricMinimum != TorqueElectricMinimum) {
        m_data.m_TorqueElectricMinimum = TorqueElectricMinimum;
        m_publisher->publishTorqueElectricMinimumChanged(TorqueElectricMinimum);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow2::getTorqueElectricMinimum() const
{
    return m_data.m_TorqueElectricMinimum;
}

void VehicleMotionManagementBrakeAxleRow2::setTorqueFrictionDifferenceMaximum(int32_t TorqueFrictionDifferenceMaximum)
{
    if (m_data.m_TorqueFrictionDifferenceMaximum != TorqueFrictionDifferenceMaximum) {
        m_data.m_TorqueFrictionDifferenceMaximum = TorqueFrictionDifferenceMaximum;
        m_publisher->publishTorqueFrictionDifferenceMaximumChanged(TorqueFrictionDifferenceMaximum);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow2::getTorqueFrictionDifferenceMaximum() const
{
    return m_data.m_TorqueFrictionDifferenceMaximum;
}

void VehicleMotionManagementBrakeAxleRow2::setTorqueDistributionFrictionRightMaximum(int32_t TorqueDistributionFrictionRightMaximum)
{
    if (m_data.m_TorqueDistributionFrictionRightMaximum != TorqueDistributionFrictionRightMaximum) {
        m_data.m_TorqueDistributionFrictionRightMaximum = TorqueDistributionFrictionRightMaximum;
        m_publisher->publishTorqueDistributionFrictionRightMaximumChanged(TorqueDistributionFrictionRightMaximum);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow2::getTorqueDistributionFrictionRightMaximum() const
{
    return m_data.m_TorqueDistributionFrictionRightMaximum;
}

void VehicleMotionManagementBrakeAxleRow2::setTorqueDistributionFrictionRightMinimum(int32_t TorqueDistributionFrictionRightMinimum)
{
    if (m_data.m_TorqueDistributionFrictionRightMinimum != TorqueDistributionFrictionRightMinimum) {
        m_data.m_TorqueDistributionFrictionRightMinimum = TorqueDistributionFrictionRightMinimum;
        m_publisher->publishTorqueDistributionFrictionRightMinimumChanged(TorqueDistributionFrictionRightMinimum);
    }
}

int32_t VehicleMotionManagementBrakeAxleRow2::getTorqueDistributionFrictionRightMinimum() const
{
    return m_data.m_TorqueDistributionFrictionRightMinimum;
}

IVehicleMotionManagementBrakeAxleRow2Publisher& VehicleMotionManagementBrakeAxleRow2::_getPublisher() const
{
    return *m_publisher;
}
