
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterystateofcharge.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain_TractionBattery_StateOfCharge implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryStateOfCharge : public IVehiclePowertrainTractionBatteryStateOfCharge
{
public:
    explicit VehiclePowertrainTractionBatteryStateOfCharge();
    ~VehiclePowertrainTractionBatteryStateOfCharge();
public:
    /**
    * Current Physical state of charge of the high voltage battery, relative to net capacity. This is not necessarily the state of charge being displayed to the customer.
    */
    void setCurrent(float Current) override;
    float getCurrent() const override;
    
    /**
    * Displayed State of charge displayed to the customer.
    */
    void setDisplayed(float Displayed) override;
    float getDisplayed() const override;
    
    /**
    * CurrentEnergy Physical state of charge of high voltage battery expressed in kWh.
    */
    void setCurrentEnergy(float CurrentEnergy) override;
    float getCurrentEnergy() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_StateOfCharge changes and signal emission.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_StateOfCharge.
    */
    IVehiclePowertrainTractionBatteryStateOfChargePublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain_TractionBattery_StateOfCharge. */
    std::unique_ptr<IVehiclePowertrainTractionBatteryStateOfChargePublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain_TractionBattery_StateOfCharge. */
    VehiclePowertrainTractionBatteryStateOfChargeData m_data;
};
} // namespace Vehicle
} // namespace Cpp
