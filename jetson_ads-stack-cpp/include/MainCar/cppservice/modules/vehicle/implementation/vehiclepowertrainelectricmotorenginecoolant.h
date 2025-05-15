
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertrainelectricmotorenginecoolant.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain_ElectricMotor_EngineCoolant implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainElectricMotorEngineCoolant : public IVehiclePowertrainElectricMotorEngineCoolant
{
public:
    explicit VehiclePowertrainElectricMotorEngineCoolant();
    ~VehiclePowertrainElectricMotorEngineCoolant();
public:
    /**
    * Capacity Engine coolant capacity in liters.
    */
    void setCapacity(float Capacity) override;
    float getCapacity() const override;
    
    /**
    * Temperature Engine coolant temperature.
    */
    void setTemperature(float Temperature) override;
    float getTemperature() const override;
    
    /**
    * Level Engine coolant level.
    */
    void setLevel(Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum Level) override;
    Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum getLevel() const override;
    
    /**
    * LifeRemaining Remaining engine coolant life in seconds. Negative values can be used to indicate that lifetime has been exceeded.
    */
    void setLifeRemaining(int32_t LifeRemaining) override;
    int32_t getLifeRemaining() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_ElectricMotor_EngineCoolant changes and signal emission.
    * @return The publisher for Vehicle_Powertrain_ElectricMotor_EngineCoolant.
    */
    IVehiclePowertrainElectricMotorEngineCoolantPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain_ElectricMotor_EngineCoolant. */
    std::unique_ptr<IVehiclePowertrainElectricMotorEngineCoolantPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain_ElectricMotor_EngineCoolant. */
    VehiclePowertrainElectricMotorEngineCoolantData m_data;
};
} // namespace Vehicle
} // namespace Cpp
