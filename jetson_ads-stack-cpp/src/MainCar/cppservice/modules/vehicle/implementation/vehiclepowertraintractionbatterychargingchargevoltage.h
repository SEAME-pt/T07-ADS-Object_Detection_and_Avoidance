
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargevoltage.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain_TractionBattery_Charging_ChargeVoltage implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingChargeVoltage : public IVehiclePowertrainTractionBatteryChargingChargeVoltage
{
public:
    explicit VehiclePowertrainTractionBatteryChargingChargeVoltage();
    ~VehiclePowertrainTractionBatteryChargingChargeVoltage();
public:
    /**
    * DC Current DC charging voltage at charging inlet.
    */
    void setDc(float DC) override;
    float getDc() const override;
    
    /**
    * Phase1 Current AC charging voltage (rms) at inlet for Phase 1.
    */
    void setPhase1(float Phase1) override;
    float getPhase1() const override;
    
    /**
    * Phase2 Current AC charging voltage (rms) at inlet for Phase 2.
    */
    void setPhase2(float Phase2) override;
    float getPhase2() const override;
    
    /**
    * Phase3 Current AC charging voltage (rms) at inlet for Phase 3.
    */
    void setPhase3(float Phase3) override;
    float getPhase3() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_ChargeVoltage changes and signal emission.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_ChargeVoltage.
    */
    IVehiclePowertrainTractionBatteryChargingChargeVoltagePublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain_TractionBattery_Charging_ChargeVoltage. */
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingChargeVoltagePublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain_TractionBattery_Charging_ChargeVoltage. */
    VehiclePowertrainTractionBatteryChargingChargeVoltageData m_data;
};
} // namespace Vehicle
} // namespace Cpp
