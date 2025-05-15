
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterycharging.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain_TractionBattery_Charging implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryCharging : public IVehiclePowertrainTractionBatteryCharging
{
public:
    explicit VehiclePowertrainTractionBatteryCharging();
    ~VehiclePowertrainTractionBatteryCharging();
public:
    /**
    * ChargeLimit Target charge limit (state of charge) for battery.
    */
    void setChargeLimit(int32_t ChargeLimit) override;
    int32_t getChargeLimit() const override;
    
    /**
    * IsCharging True if charging is ongoing. Charging is considered to be ongoing if energy is flowing from charger to vehicle.
    */
    void setIsCharging(bool IsCharging) override;
    bool getIsCharging() const override;
    
    /**
    * IsDischarging True if discharging (vehicle to grid) is ongoing. Discharging is considered to be ongoing if energy is flowing from vehicle to charger/grid.
    */
    void setIsDischarging(bool IsDischarging) override;
    bool getIsDischarging() const override;
    
    /**
    * StartStopCharging Start or stop the charging process.
    */
    void setStartStopCharging(Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum StartStopCharging) override;
    Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum getStartStopCharging() const override;
    
    /**
    * AveragePower Average charging power of last or current charging event.
    */
    void setAveragePower(float AveragePower) override;
    float getAveragePower() const override;
    
    /**
    * MaxPower Maximum charging power of last or current charging event.
    */
    void setMaxPower(float MaxPower) override;
    float getMaxPower() const override;
    
    /**
    * EvseId EVSE charging point ID (without separators) of last or current charging event according to ISO 15118-2 Annex H.
    */
    void setEvseId(const std::string& EvseId) override;
    const std::string& getEvseId() const override;
    
    /**
    * PowerLoss Electrical energy lost by power dissipation to heat inside the AC/DC converter.
    */
    void setPowerLoss(float PowerLoss) override;
    float getPowerLoss() const override;
    
    /**
    * Temperature Current temperature of AC/DC converter converting grid voltage to battery voltage.
    */
    void setTemperature(float Temperature) override;
    float getTemperature() const override;
    
    /**
    * ChargeRate Current charging rate, as in kilometers of range added per hour.
    */
    void setChargeRate(float ChargeRate) override;
    float getChargeRate() const override;
    
    /**
    * TimeToComplete The time needed for the current charging process to reach Charging.ChargeLimit. 0 if charging is complete or no charging process is active or planned.
    */
    void setTimeToComplete(int32_t TimeToComplete) override;
    int32_t getTimeToComplete() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging changes and signal emission.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging.
    */
    IVehiclePowertrainTractionBatteryChargingPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain_TractionBattery_Charging. */
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain_TractionBattery_Charging. */
    VehiclePowertrainTractionBatteryChargingData m_data;
};
} // namespace Vehicle
} // namespace Cpp
