
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingchargingportfrontmiddle.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle : public IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle
{
public:
    explicit VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle();
    ~VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddle();
public:
    /**
    * SupportedInletTypes A list of the supported (i.e., available) charging inlets in a particular charging port. IEC types refer to IEC 62196,  GBT refers to  GB/T 20234.
    */
    void setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum SupportedInletTypes) override;
    Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum getSupportedInletTypes() const override;
    
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
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle changes and signal emission.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle.
    */
    IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle. */
    std::unique_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortFrontMiddlePublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle. */
    VehiclePowertrainTractionBatteryChargingChargingPortFrontMiddleData m_data;
};
} // namespace Vehicle
} // namespace Cpp
