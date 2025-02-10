
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargecurrent.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingChargeCurrent : public IVehiclePowertrainTractionBatteryChargingChargeCurrent
{
public:
    explicit VehiclePowertrainTractionBatteryChargingChargeCurrent();
    ~VehiclePowertrainTractionBatteryChargingChargeCurrent();
public:
    /**
    * DC Current DC charging current at inlet. Negative if returning energy to grid.
    */
    void setDc(float DC) override;
    float getDc() const override;
    
    /**
    * Phase1 Current AC charging current (rms) at inlet for Phase 1. Negative if returning energy to grid.
    */
    void setPhase1(float Phase1) override;
    float getPhase1() const override;
    
    /**
    * Phase2 Current AC charging current (rms) at inlet for Phase 2. Negative if returning energy to grid.
    */
    void setPhase2(float Phase2) override;
    float getPhase2() const override;
    
    /**
    * Phase3 Current AC charging current (rms) at inlet for Phase 3. Negative if returning energy to grid.
    */
    void setPhase3(float Phase3) override;
    float getPhase3() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent changes and signal emission.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent.
    */
    IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent. */
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent. */
    VehiclePowertrainTractionBatteryChargingChargeCurrentData m_data;
};
} // namespace Vehicle
} // namespace Cpp
