
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertrainelectricmotor.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain_ElectricMotor implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainElectricMotor : public IVehiclePowertrainElectricMotor
{
public:
    explicit VehiclePowertrainElectricMotor();
    ~VehiclePowertrainElectricMotor();
public:
    /**
    * EngineCode Engine code designation, as specified by vehicle manufacturer.
    */
    void setEngineCode(const std::string& EngineCode) override;
    const std::string& getEngineCode() const override;
    
    /**
    * MaxPower Peak power, in kilowatts, that motor(s) can generate.
    */
    void setMaxPower(int32_t MaxPower) override;
    int32_t getMaxPower() const override;
    
    /**
    * MaxTorque Peak power, in newton meter, that the motor(s) can generate.
    */
    void setMaxTorque(int32_t MaxTorque) override;
    int32_t getMaxTorque() const override;
    
    /**
    * MaxRegenPower Peak regen/brake power, in kilowatts, that motor(s) can generate.
    */
    void setMaxRegenPower(int32_t MaxRegenPower) override;
    int32_t getMaxRegenPower() const override;
    
    /**
    * MaxRegenTorque Peak regen/brake torque, in newton meter, that the motor(s) can generate.
    */
    void setMaxRegenTorque(int32_t MaxRegenTorque) override;
    int32_t getMaxRegenTorque() const override;
    
    /**
    * Speed Motor rotational speed measured as rotations per minute. Negative values indicate reverse driving mode.
    */
    void setSpeed(int32_t Speed) override;
    int32_t getSpeed() const override;
    
    /**
    * TimeInUse Accumulated time during engine lifetime when the vehicule state's is 'READY'.
    */
    void setTimeInUse(float TimeInUse) override;
    float getTimeInUse() const override;
    
    /**
    * Temperature Motor temperature.
    */
    void setTemperature(float Temperature) override;
    float getTemperature() const override;
    
    /**
    * CoolantTemperature_Deprecated Motor coolant temperature (if applicable).
    */
    void setCoolantTemperatureDeprecated(float CoolantTemperature_Deprecated) override;
    float getCoolantTemperatureDeprecated() const override;
    
    /**
    * Power Current motor power output. Negative values indicate regen mode.
    */
    void setPower(int32_t Power) override;
    int32_t getPower() const override;
    
    /**
    * Torque Current motor torque. Negative values indicate regen mode.
    */
    void setTorque(int32_t Torque) override;
    int32_t getTorque() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_ElectricMotor changes and signal emission.
    * @return The publisher for Vehicle_Powertrain_ElectricMotor.
    */
    IVehiclePowertrainElectricMotorPublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain_ElectricMotor. */
    std::unique_ptr<IVehiclePowertrainElectricMotorPublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain_ElectricMotor. */
    VehiclePowertrainElectricMotorData m_data;
};
} // namespace Vehicle
} // namespace Cpp
