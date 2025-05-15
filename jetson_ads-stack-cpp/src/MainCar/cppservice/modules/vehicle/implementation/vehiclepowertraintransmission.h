
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertraintransmission.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain_Transmission implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTransmission : public IVehiclePowertrainTransmission
{
public:
    explicit VehiclePowertrainTransmission();
    ~VehiclePowertrainTransmission();
public:
    /**
    * Type Transmission type.
    */
    void setType(Vehicle_Powertrain_Transmission_Type_ValueEnum Type) override;
    Vehicle_Powertrain_Transmission_Type_ValueEnum getType() const override;
    
    /**
    * GearCount Number of forward gears in the transmission. -1 = CVT.
    */
    void setGearCount(int32_t GearCount) override;
    int32_t getGearCount() const override;
    
    /**
    * DriveType Drive type.
    */
    void setDriveType(Vehicle_Powertrain_Transmission_DriveType_ValueEnum DriveType) override;
    Vehicle_Powertrain_Transmission_DriveType_ValueEnum getDriveType() const override;
    
    /**
    * TravelledDistance Odometer reading, total distance travelled during the lifetime of the transmission.
    */
    void setTravelledDistance(float TravelledDistance) override;
    float getTravelledDistance() const override;
    
    /**
    * CurrentGear The current gear. 0=Neutral, 1/2/..=Forward, -1/-2/..=Reverse.
    */
    void setCurrentGear(int32_t CurrentGear) override;
    int32_t getCurrentGear() const override;
    
    /**
    * SelectedGear The selected gear. 0=Neutral, 1/2/..=Forward, -1/-2/..=Reverse, 126=Park, 127=Drive.
    */
    void setSelectedGear(int32_t SelectedGear) override;
    int32_t getSelectedGear() const override;
    
    /**
    * IsParkLockEngaged Is the transmission park lock engaged or not. False = Disengaged. True = Engaged.
    */
    void setIsParkLockEngaged(bool IsParkLockEngaged) override;
    bool getIsParkLockEngaged() const override;
    
    /**
    * IsLowRangeEngaged Is gearbox in low range mode or not. False = Normal/High range engaged. True = Low range engaged.
    */
    void setIsLowRangeEngaged(bool IsLowRangeEngaged) override;
    bool getIsLowRangeEngaged() const override;
    
    /**
    * IsElectricalPowertrainEngaged Is electrical powertrain mechanically connected/engaged to the drivetrain or not. False = Disconnected/Disengaged. True = Connected/Engaged.
    */
    void setIsElectricalPowertrainEngaged(bool IsElectricalPowertrainEngaged) override;
    bool getIsElectricalPowertrainEngaged() const override;
    
    /**
    * PerformanceMode Current gearbox performance mode.
    */
    void setPerformanceMode(Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum PerformanceMode) override;
    Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum getPerformanceMode() const override;
    
    /**
    * GearChangeMode Is the gearbox in automatic or manual (paddle) mode.
    */
    void setGearChangeMode(Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum GearChangeMode) override;
    Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum getGearChangeMode() const override;
    
    /**
    * Temperature The current gearbox temperature.
    */
    void setTemperature(float Temperature) override;
    float getTemperature() const override;
    
    /**
    * ClutchEngagement Clutch engagement. 0% = Clutch fully disengaged. 100% = Clutch fully engaged.
    */
    void setClutchEngagement(float ClutchEngagement) override;
    float getClutchEngagement() const override;
    
    /**
    * ClutchWear Clutch wear as a percent. 0 = no wear. 100 = worn.
    */
    void setClutchWear(int32_t ClutchWear) override;
    int32_t getClutchWear() const override;
    
    /**
    * DiffLockFrontEngagement Front Diff Lock engagement. 0% = Diff lock fully disengaged. 100% = Diff lock fully engaged.
    */
    void setDiffLockFrontEngagement(float DiffLockFrontEngagement) override;
    float getDiffLockFrontEngagement() const override;
    
    /**
    * DiffLockRearEngagement Rear Diff Lock engagement. 0% = Diff lock fully disengaged. 100% = Diff lock fully engaged.
    */
    void setDiffLockRearEngagement(float DiffLockRearEngagement) override;
    float getDiffLockRearEngagement() const override;
    
    /**
    * TorqueDistribution Torque distribution between front and rear axle in percent. -100% = Full torque to front axle, 0% = 50:50 Front/Rear, 100% = Full torque to rear axle.
    */
    void setTorqueDistribution(float TorqueDistribution) override;
    float getTorqueDistribution() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_Transmission changes and signal emission.
    * @return The publisher for Vehicle_Powertrain_Transmission.
    */
    IVehiclePowertrainTransmissionPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain_Transmission. */
    std::unique_ptr<IVehiclePowertrainTransmissionPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain_Transmission. */
    VehiclePowertrainTransmissionData m_data;
};
} // namespace Vehicle
} // namespace Cpp
