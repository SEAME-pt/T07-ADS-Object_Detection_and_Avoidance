

#include "vehicle/implementation/vehiclemotionmanagementbrake.h"
#include "vehicle/generated/core/vehiclemotionmanagementbrake.publisher.h"
#include "vehicle/generated/core/vehiclemotionmanagementbrake.data.h"

using namespace Cpp::Vehicle;

VehicleMotionManagementBrake::VehicleMotionManagementBrake()
    : m_publisher(std::make_unique<VehicleMotionManagementBrakePublisher>())
{
}
VehicleMotionManagementBrake::~VehicleMotionManagementBrake()
{
}

void VehicleMotionManagementBrake::setVehicleForceMaximum(int32_t VehicleForceMaximum)
{
    if (m_data.m_VehicleForceMaximum != VehicleForceMaximum) {
        m_data.m_VehicleForceMaximum = VehicleForceMaximum;
        m_publisher->publishVehicleForceMaximumChanged(VehicleForceMaximum);
    }
}

int32_t VehicleMotionManagementBrake::getVehicleForceMaximum() const
{
    return m_data.m_VehicleForceMaximum;
}

void VehicleMotionManagementBrake::setVehicleForceElectricMinimumArbitrated(int32_t VehicleForceElectricMinimumArbitrated)
{
    if (m_data.m_VehicleForceElectricMinimumArbitrated != VehicleForceElectricMinimumArbitrated) {
        m_data.m_VehicleForceElectricMinimumArbitrated = VehicleForceElectricMinimumArbitrated;
        m_publisher->publishVehicleForceElectricMinimumArbitratedChanged(VehicleForceElectricMinimumArbitrated);
    }
}

int32_t VehicleMotionManagementBrake::getVehicleForceElectricMinimumArbitrated() const
{
    return m_data.m_VehicleForceElectricMinimumArbitrated;
}

void VehicleMotionManagementBrake::setVehicleForceElectric(int32_t VehicleForceElectric)
{
    if (m_data.m_VehicleForceElectric != VehicleForceElectric) {
        m_data.m_VehicleForceElectric = VehicleForceElectric;
        m_publisher->publishVehicleForceElectricChanged(VehicleForceElectric);
    }
}

int32_t VehicleMotionManagementBrake::getVehicleForceElectric() const
{
    return m_data.m_VehicleForceElectric;
}

void VehicleMotionManagementBrake::setVehicleForceDistributionFrontMaximum(int32_t VehicleForceDistributionFrontMaximum)
{
    if (m_data.m_VehicleForceDistributionFrontMaximum != VehicleForceDistributionFrontMaximum) {
        m_data.m_VehicleForceDistributionFrontMaximum = VehicleForceDistributionFrontMaximum;
        m_publisher->publishVehicleForceDistributionFrontMaximumChanged(VehicleForceDistributionFrontMaximum);
    }
}

int32_t VehicleMotionManagementBrake::getVehicleForceDistributionFrontMaximum() const
{
    return m_data.m_VehicleForceDistributionFrontMaximum;
}

void VehicleMotionManagementBrake::setVehicleForceDistributionFrontMinimum(int32_t VehicleForceDistributionFrontMinimum)
{
    if (m_data.m_VehicleForceDistributionFrontMinimum != VehicleForceDistributionFrontMinimum) {
        m_data.m_VehicleForceDistributionFrontMinimum = VehicleForceDistributionFrontMinimum;
        m_publisher->publishVehicleForceDistributionFrontMinimumChanged(VehicleForceDistributionFrontMinimum);
    }
}

int32_t VehicleMotionManagementBrake::getVehicleForceDistributionFrontMinimum() const
{
    return m_data.m_VehicleForceDistributionFrontMinimum;
}

IVehicleMotionManagementBrakePublisher& VehicleMotionManagementBrake::_getPublisher() const
{
    return *m_publisher;
}
