
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertraintractionbattery.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain_TractionBattery implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBattery : public IVehiclePowertrainTractionBattery
{
public:
    explicit VehiclePowertrainTractionBattery();
    ~VehiclePowertrainTractionBattery();
public:
    /**
    * Id Battery Identification Number as assigned by OEM.
    */
    void setId(const std::string& Id) override;
    const std::string& getId() const override;
    
    /**
    * ProductionDate Production date of battery in ISO8601 format, e.g. YYYY-MM-DD.
    */
    void setProductionDate(const std::string& ProductionDate) override;
    const std::string& getProductionDate() const override;
    
    /**
    * ErrorCodes Current error codes related to the battery, if any.
    */
    void setErrorCodes(const std::list<std::string>& ErrorCodes) override;
    const std::list<std::string>& getErrorCodes() const override;
    
    /**
    * IsPowerConnected Indicating if the power (positive terminator) of the traction battery is connected to the powertrain.
    */
    void setIsPowerConnected(bool IsPowerConnected) override;
    bool getIsPowerConnected() const override;
    
    /**
    * IsGroundConnected Indicating if the ground (negative terminator) of the traction battery is connected to the powertrain.
    */
    void setIsGroundConnected(bool IsGroundConnected) override;
    bool getIsGroundConnected() const override;
    
    /**
    * GrossCapacity Gross capacity of the battery.
    */
    void setGrossCapacity(int32_t GrossCapacity) override;
    int32_t getGrossCapacity() const override;
    
    /**
    * NetCapacity Total net capacity of the battery considering aging.
    */
    void setNetCapacity(int32_t NetCapacity) override;
    int32_t getNetCapacity() const override;
    
    /**
    * StateOfHealth Calculated battery state of health at standard conditions.
    */
    void setStateOfHealth(float StateOfHealth) override;
    float getStateOfHealth() const override;
    
    /**
    * NominalVoltage Nominal Voltage of the battery.
    */
    void setNominalVoltage(int32_t NominalVoltage) override;
    int32_t getNominalVoltage() const override;
    
    /**
    * MaxVoltage Max allowed voltage of the battery, e.g. during charging.
    */
    void setMaxVoltage(int32_t MaxVoltage) override;
    int32_t getMaxVoltage() const override;
    
    /**
    * CurrentVoltage Current Voltage of the battery.
    */
    void setCurrentVoltage(float CurrentVoltage) override;
    float getCurrentVoltage() const override;
    
    /**
    * CurrentCurrent Current current flowing in/out of battery. Positive = Current flowing in to battery, e.g. during charging. Negative = Current flowing out of battery, e.g. during driving.
    */
    void setCurrentCurrent(float CurrentCurrent) override;
    float getCurrentCurrent() const override;
    
    /**
    * CurrentPower Current electrical energy flowing in/out of battery. Positive = Energy flowing in to battery, e.g. during charging. Negative = Energy flowing out of battery, e.g. during driving.
    */
    void setCurrentPower(float CurrentPower) override;
    float getCurrentPower() const override;
    
    /**
    * AccumulatedChargedEnergy The accumulated energy delivered to the battery during charging over lifetime of the battery.
    */
    void setAccumulatedChargedEnergy(float AccumulatedChargedEnergy) override;
    float getAccumulatedChargedEnergy() const override;
    
    /**
    * AccumulatedConsumedEnergy The accumulated energy leaving HV battery for propulsion and auxiliary loads over lifetime of the battery.
    */
    void setAccumulatedConsumedEnergy(float AccumulatedConsumedEnergy) override;
    float getAccumulatedConsumedEnergy() const override;
    
    /**
    * AccumulatedChargedThroughput The accumulated charge throughput delivered to the battery during charging over lifetime of the battery.
    */
    void setAccumulatedChargedThroughput(float AccumulatedChargedThroughput) override;
    float getAccumulatedChargedThroughput() const override;
    
    /**
    * AccumulatedConsumedThroughput The accumulated charge throughput leaving HV battery for propulsion and auxiliary loads over lifetime of the battery.
    */
    void setAccumulatedConsumedThroughput(float AccumulatedConsumedThroughput) override;
    float getAccumulatedConsumedThroughput() const override;
    
    /**
    * PowerLoss Electrical energy lost by power dissipation to heat inside the battery.
    */
    void setPowerLoss(float PowerLoss) override;
    float getPowerLoss() const override;
    
    /**
    * Range Remaining range in meters using only battery.
    */
    void setRange(int32_t Range) override;
    int32_t getRange() const override;
    
    /**
    * TimeRemaining Time remaining in seconds before the battery is empty.
    */
    void setTimeRemaining(int32_t TimeRemaining) override;
    int32_t getTimeRemaining() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery changes and signal emission.
    * @return The publisher for Vehicle_Powertrain_TractionBattery.
    */
    IVehiclePowertrainTractionBatteryPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain_TractionBattery. */
    std::unique_ptr<IVehiclePowertrainTractionBatteryPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain_TractionBattery. */
    VehiclePowertrainTractionBatteryData m_data;
};
} // namespace Vehicle
} // namespace Cpp
