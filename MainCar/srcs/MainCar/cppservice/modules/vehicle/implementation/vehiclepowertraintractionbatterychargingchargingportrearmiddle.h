
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportrearmiddle.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle : public IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddle
{
public:
    explicit VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle();
    ~VehiclePowertrainTractionBatteryChargingChargingPortRearMiddle();
public:
    /**
    * SupportedInletTypes A list of the supported (i.e., available) charging inlets in a particular charging port. IEC types refer to IEC 62196,  GBT refers to  GB/T 20234.
    */
    void setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum SupportedInletTypes) override;
    Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum getSupportedInletTypes() const override;
    
    /**
    * IsFlapOpen Status of the charging port flap(s).
    */
    void setIsFlapOpen(bool IsFlapOpen) override;
    bool getIsFlapOpen() const override;
    
    /**
    * IsChargingCableConnected Indicates whether a charging cable is physically connected to a particular charging port or not.
    */
    void setIsChargingCableConnected(bool IsChargingCableConnected) override;
    bool getIsChargingCableConnected() const override;
    
    /**
    * IsChargingCableLocked Is charging cable locked to prevent removal.
    */
    void setIsChargingCableLocked(bool IsChargingCableLocked) override;
    bool getIsChargingCableLocked() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle changes and signal emission.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle.
    */
    IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle. */
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortRearMiddlePublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle. */
    VehiclePowertrainTractionBatteryChargingChargingPortRearMiddleData m_data;
};
} // namespace Vehicle
} // namespace Cpp
