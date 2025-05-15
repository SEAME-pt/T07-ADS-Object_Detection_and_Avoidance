

#include "vehicle/implementation/vehiclepowertraintransmission.h"
#include "vehicle/generated/core/vehiclepowertraintransmission.publisher.h"
#include "vehicle/generated/core/vehiclepowertraintransmission.data.h"

using namespace Cpp::Vehicle;

VehiclePowertrainTransmission::VehiclePowertrainTransmission()
    : m_publisher(std::make_unique<VehiclePowertrainTransmissionPublisher>())
{
}
VehiclePowertrainTransmission::~VehiclePowertrainTransmission()
{
}

void VehiclePowertrainTransmission::setType(Vehicle_Powertrain_Transmission_Type_ValueEnum Type)
{
    if (m_data.m_Type != Type) {
        m_data.m_Type = Type;
        m_publisher->publishTypeChanged(Type);
    }
}

Vehicle_Powertrain_Transmission_Type_ValueEnum VehiclePowertrainTransmission::getType() const
{
    return m_data.m_Type;
}

void VehiclePowertrainTransmission::setGearCount(int32_t GearCount)
{
    if (m_data.m_GearCount != GearCount) {
        m_data.m_GearCount = GearCount;
        m_publisher->publishGearCountChanged(GearCount);
    }
}

int32_t VehiclePowertrainTransmission::getGearCount() const
{
    return m_data.m_GearCount;
}

void VehiclePowertrainTransmission::setDriveType(Vehicle_Powertrain_Transmission_DriveType_ValueEnum DriveType)
{
    if (m_data.m_DriveType != DriveType) {
        m_data.m_DriveType = DriveType;
        m_publisher->publishDriveTypeChanged(DriveType);
    }
}

Vehicle_Powertrain_Transmission_DriveType_ValueEnum VehiclePowertrainTransmission::getDriveType() const
{
    return m_data.m_DriveType;
}

void VehiclePowertrainTransmission::setTravelledDistance(float TravelledDistance)
{
    if (m_data.m_TravelledDistance != TravelledDistance) {
        m_data.m_TravelledDistance = TravelledDistance;
        m_publisher->publishTravelledDistanceChanged(TravelledDistance);
    }
}

float VehiclePowertrainTransmission::getTravelledDistance() const
{
    return m_data.m_TravelledDistance;
}

void VehiclePowertrainTransmission::setCurrentGear(int32_t CurrentGear)
{
    if (m_data.m_CurrentGear != CurrentGear) {
        m_data.m_CurrentGear = CurrentGear;
        m_publisher->publishCurrentGearChanged(CurrentGear);
    }
}

int32_t VehiclePowertrainTransmission::getCurrentGear() const
{
    return m_data.m_CurrentGear;
}

void VehiclePowertrainTransmission::setSelectedGear(int32_t SelectedGear)
{
    if (m_data.m_SelectedGear != SelectedGear) {
        m_data.m_SelectedGear = SelectedGear;
        m_publisher->publishSelectedGearChanged(SelectedGear);
    }
}

int32_t VehiclePowertrainTransmission::getSelectedGear() const
{
    return m_data.m_SelectedGear;
}

void VehiclePowertrainTransmission::setIsParkLockEngaged(bool IsParkLockEngaged)
{
    if (m_data.m_IsParkLockEngaged != IsParkLockEngaged) {
        m_data.m_IsParkLockEngaged = IsParkLockEngaged;
        m_publisher->publishIsParkLockEngagedChanged(IsParkLockEngaged);
    }
}

bool VehiclePowertrainTransmission::getIsParkLockEngaged() const
{
    return m_data.m_IsParkLockEngaged;
}

void VehiclePowertrainTransmission::setIsLowRangeEngaged(bool IsLowRangeEngaged)
{
    if (m_data.m_IsLowRangeEngaged != IsLowRangeEngaged) {
        m_data.m_IsLowRangeEngaged = IsLowRangeEngaged;
        m_publisher->publishIsLowRangeEngagedChanged(IsLowRangeEngaged);
    }
}

bool VehiclePowertrainTransmission::getIsLowRangeEngaged() const
{
    return m_data.m_IsLowRangeEngaged;
}

void VehiclePowertrainTransmission::setIsElectricalPowertrainEngaged(bool IsElectricalPowertrainEngaged)
{
    if (m_data.m_IsElectricalPowertrainEngaged != IsElectricalPowertrainEngaged) {
        m_data.m_IsElectricalPowertrainEngaged = IsElectricalPowertrainEngaged;
        m_publisher->publishIsElectricalPowertrainEngagedChanged(IsElectricalPowertrainEngaged);
    }
}

bool VehiclePowertrainTransmission::getIsElectricalPowertrainEngaged() const
{
    return m_data.m_IsElectricalPowertrainEngaged;
}

void VehiclePowertrainTransmission::setPerformanceMode(Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum PerformanceMode)
{
    if (m_data.m_PerformanceMode != PerformanceMode) {
        m_data.m_PerformanceMode = PerformanceMode;
        m_publisher->publishPerformanceModeChanged(PerformanceMode);
    }
}

Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum VehiclePowertrainTransmission::getPerformanceMode() const
{
    return m_data.m_PerformanceMode;
}

void VehiclePowertrainTransmission::setGearChangeMode(Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum GearChangeMode)
{
    if (m_data.m_GearChangeMode != GearChangeMode) {
        m_data.m_GearChangeMode = GearChangeMode;
        m_publisher->publishGearChangeModeChanged(GearChangeMode);
    }
}

Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum VehiclePowertrainTransmission::getGearChangeMode() const
{
    return m_data.m_GearChangeMode;
}

void VehiclePowertrainTransmission::setTemperature(float Temperature)
{
    if (m_data.m_Temperature != Temperature) {
        m_data.m_Temperature = Temperature;
        m_publisher->publishTemperatureChanged(Temperature);
    }
}

float VehiclePowertrainTransmission::getTemperature() const
{
    return m_data.m_Temperature;
}

void VehiclePowertrainTransmission::setClutchEngagement(float ClutchEngagement)
{
    if (m_data.m_ClutchEngagement != ClutchEngagement) {
        m_data.m_ClutchEngagement = ClutchEngagement;
        m_publisher->publishClutchEngagementChanged(ClutchEngagement);
    }
}

float VehiclePowertrainTransmission::getClutchEngagement() const
{
    return m_data.m_ClutchEngagement;
}

void VehiclePowertrainTransmission::setClutchWear(int32_t ClutchWear)
{
    if (m_data.m_ClutchWear != ClutchWear) {
        m_data.m_ClutchWear = ClutchWear;
        m_publisher->publishClutchWearChanged(ClutchWear);
    }
}

int32_t VehiclePowertrainTransmission::getClutchWear() const
{
    return m_data.m_ClutchWear;
}

void VehiclePowertrainTransmission::setDiffLockFrontEngagement(float DiffLockFrontEngagement)
{
    if (m_data.m_DiffLockFrontEngagement != DiffLockFrontEngagement) {
        m_data.m_DiffLockFrontEngagement = DiffLockFrontEngagement;
        m_publisher->publishDiffLockFrontEngagementChanged(DiffLockFrontEngagement);
    }
}

float VehiclePowertrainTransmission::getDiffLockFrontEngagement() const
{
    return m_data.m_DiffLockFrontEngagement;
}

void VehiclePowertrainTransmission::setDiffLockRearEngagement(float DiffLockRearEngagement)
{
    if (m_data.m_DiffLockRearEngagement != DiffLockRearEngagement) {
        m_data.m_DiffLockRearEngagement = DiffLockRearEngagement;
        m_publisher->publishDiffLockRearEngagementChanged(DiffLockRearEngagement);
    }
}

float VehiclePowertrainTransmission::getDiffLockRearEngagement() const
{
    return m_data.m_DiffLockRearEngagement;
}

void VehiclePowertrainTransmission::setTorqueDistribution(float TorqueDistribution)
{
    if (m_data.m_TorqueDistribution != TorqueDistribution) {
        m_data.m_TorqueDistribution = TorqueDistribution;
        m_publisher->publishTorqueDistributionChanged(TorqueDistribution);
    }
}

float VehiclePowertrainTransmission::getTorqueDistribution() const
{
    return m_data.m_TorqueDistribution;
}

IVehiclePowertrainTransmissionPublisher& VehiclePowertrainTransmission::_getPublisher() const
{
    return *m_publisher;
}
