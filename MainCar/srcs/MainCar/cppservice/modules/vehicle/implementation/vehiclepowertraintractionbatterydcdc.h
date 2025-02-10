
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterydcdc.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain_TractionBattery_DCDC implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryDcdc : public IVehiclePowertrainTractionBatteryDcdc
{
public:
    explicit VehiclePowertrainTractionBatteryDcdc();
    ~VehiclePowertrainTractionBatteryDcdc();
public:
    /**
    * PowerLoss Electrical energy lost by power dissipation to heat inside DC/DC converter.
    */
    void setPowerLoss(float PowerLoss) override;
    float getPowerLoss() const override;
    
    /**
    * Temperature Current temperature of DC/DC converter converting battery high voltage to vehicle low voltage (typically 12 Volts).
    */
    void setTemperature(float Temperature) override;
    float getTemperature() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_DCDC changes and signal emission.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_DCDC.
    */
    IVehiclePowertrainTractionBatteryDcdcPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain_TractionBattery_DCDC. */
    std::unique_ptr<IVehiclePowertrainTractionBatteryDcdcPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain_TractionBattery_DCDC. */
    VehiclePowertrainTractionBatteryDcdcData m_data;
};
} // namespace Vehicle
} // namespace Cpp
