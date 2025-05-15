
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterybatteryconditioning.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain_TractionBattery_BatteryConditioning implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryBatteryConditioning : public IVehiclePowertrainTractionBatteryBatteryConditioning
{
public:
    explicit VehiclePowertrainTractionBatteryBatteryConditioning();
    ~VehiclePowertrainTractionBatteryBatteryConditioning();
public:
    /**
    * IsActive Indicates if battery conditioning is active (i.e. actively monitors battery temperature). True = Active. False = Inactive.
    */
    void setIsActive(bool IsActive) override;
    bool getIsActive() const override;
    
    /**
    * IsOngoing Indicating if battery conditioning is currently ongoing. Battery conditioning is considered ongoing when the battery conditioning system is actively heating or cooling the battery, or requesting heating or cooling.
    */
    void setIsOngoing(bool IsOngoing) override;
    bool getIsOngoing() const override;
    
    /**
    * RequestedMode Defines requested mode for battery conditioning. INACTIVE - Battery conditioning inactive. FAST_CHARGING_PREPARATION - Battery conditioning for fast charging. DRIVING_PREPARATION - Battery conditioning for driving.
    */
    void setRequestedMode(Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum RequestedMode) override;
    Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum getRequestedMode() const override;
    
    /**
    * StartTime Start time for battery conditioning, formatted according to ISO 8601 with UTC time zone.
    */
    void setStartTime(const std::string& StartTime) override;
    const std::string& getStartTime() const override;
    
    /**
    * TargetTime Target time when conditioning shall be finished, formatted according to ISO 8601 with UTC time zone.
    */
    void setTargetTime(const std::string& TargetTime) override;
    const std::string& getTargetTime() const override;
    
    /**
    * TargetTemperature Target temperature for battery conditioning.
    */
    void setTargetTemperature(float TargetTemperature) override;
    float getTargetTemperature() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_BatteryConditioning changes and signal emission.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_BatteryConditioning.
    */
    IVehiclePowertrainTractionBatteryBatteryConditioningPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain_TractionBattery_BatteryConditioning. */
    std::unique_ptr<IVehiclePowertrainTractionBatteryBatteryConditioningPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain_TractionBattery_BatteryConditioning. */
    VehiclePowertrainTractionBatteryBatteryConditioningData m_data;
};
} // namespace Vehicle
} // namespace Cpp
