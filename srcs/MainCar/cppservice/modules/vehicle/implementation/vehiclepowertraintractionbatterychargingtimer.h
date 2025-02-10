
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingtimer.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain_TractionBattery_Charging_Timer implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingTimer : public IVehiclePowertrainTractionBatteryChargingTimer
{
public:
    explicit VehiclePowertrainTractionBatteryChargingTimer();
    ~VehiclePowertrainTractionBatteryChargingTimer();
public:
    /**
    * Mode Defines timer mode for charging: INACTIVE - no timer set, charging may start as soon as battery is connected to a charger. START_TIME - charging shall start at Charging.Timer.Time. END_TIME - charging shall be finished (reach Charging.ChargeLimit) at Charging.Timer.Time. When charging is completed the vehicle shall change mode to 'inactive' or set a new Charging.Timer.Time. Charging shall start immediately if mode is 'starttime' or 'endtime' and Charging.Timer.Time is a time in the past.
    */
    void setMode(Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum Mode) override;
    Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum getMode() const override;
    
    /**
    * Time Time for next charging-related action, formatted according to ISO 8601 with UTC time zone. Value has no significance if Charging.Timer.Mode is 'inactive'.
    */
    void setTime(const std::string& Time) override;
    const std::string& getTime() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_Timer changes and signal emission.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_Timer.
    */
    IVehiclePowertrainTractionBatteryChargingTimerPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain_TractionBattery_Charging_Timer. */
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingTimerPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain_TractionBattery_Charging_Timer. */
    VehiclePowertrainTractionBatteryChargingTimerData m_data;
};
} // namespace Vehicle
} // namespace Cpp
