
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingmaximumchargingcurrent.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent : public IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrent
{
public:
    explicit VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent();
    ~VehiclePowertrainTractionBatteryChargingMaximumChargingCurrent();
public:
    /**
    * DC Maximum DC charging current at inlet that can be accepted by the system.
    */
    void setDc(float DC) override;
    float getDc() const override;
    
    /**
    * Phase1 Maximum AC charging current (rms) at inlet for Phase 1 that can be accepted by the system.
    */
    void setPhase1(float Phase1) override;
    float getPhase1() const override;
    
    /**
    * Phase2 Maximum AC charging current (rms) at inlet for Phase 2 that can be accepted by the system.
    */
    void setPhase2(float Phase2) override;
    float getPhase2() const override;
    
    /**
    * Phase3 Maximum AC charging current (rms) at inlet for Phase 3 that can be accepted by the system.
    */
    void setPhase3(float Phase3) override;
    float getPhase3() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent changes and signal emission.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent.
    */
    IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent. */
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain_TractionBattery_Charging_MaximumChargingCurrent. */
    VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentData m_data;
};
} // namespace Vehicle
} // namespace Cpp
