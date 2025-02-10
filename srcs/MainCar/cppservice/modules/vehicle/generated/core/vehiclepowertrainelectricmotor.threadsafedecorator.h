
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehiclePowertrainElectricMotorThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Powertrain_ElectricMotor interface.
* @see Vehicle_Powertrain_ElectricMotor
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehiclePowertrainElectricMotor> testVehiclePowertrainElectricMotor = std::make_unique<VehiclePowertrainElectricMotorThreadSafeDecorator>(std::make_shared<VehiclePowertrainElectricMotor>());

// Thread safe access
auto engineCode = testVehiclePowertrainElectricMotor->getEngineCode();
testVehiclePowertrainElectricMotor->setEngineCode(std::string());
auto maxPower = testVehiclePowertrainElectricMotor->getMaxPower();
testVehiclePowertrainElectricMotor->setMaxPower(0);
auto maxTorque = testVehiclePowertrainElectricMotor->getMaxTorque();
testVehiclePowertrainElectricMotor->setMaxTorque(0);
auto maxRegenPower = testVehiclePowertrainElectricMotor->getMaxRegenPower();
testVehiclePowertrainElectricMotor->setMaxRegenPower(0);
auto maxRegenTorque = testVehiclePowertrainElectricMotor->getMaxRegenTorque();
testVehiclePowertrainElectricMotor->setMaxRegenTorque(0);
auto speed = testVehiclePowertrainElectricMotor->getSpeed();
testVehiclePowertrainElectricMotor->setSpeed(0);
auto timeInUse = testVehiclePowertrainElectricMotor->getTimeInUse();
testVehiclePowertrainElectricMotor->setTimeInUse(0.0f);
auto temperature = testVehiclePowertrainElectricMotor->getTemperature();
testVehiclePowertrainElectricMotor->setTemperature(0.0f);
auto coolantTemperatureDeprecated = testVehiclePowertrainElectricMotor->getCoolantTemperatureDeprecated();
testVehiclePowertrainElectricMotor->setCoolantTemperatureDeprecated(0.0f);
auto power = testVehiclePowertrainElectricMotor->getPower();
testVehiclePowertrainElectricMotor->setPower(0);
auto torque = testVehiclePowertrainElectricMotor->getTorque();
testVehiclePowertrainElectricMotor->setTorque(0);
\endcode
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainElectricMotorThreadSafeDecorator : public IVehiclePowertrainElectricMotor
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Powertrain_ElectricMotor object to make thread safe.
    */
    explicit VehiclePowertrainElectricMotorThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainElectricMotor> impl);

    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    void setEngineCode(const std::string& EngineCode) override;
    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    const std::string& getEngineCode() const override;

    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    void setMaxPower(int32_t MaxPower) override;
    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    int32_t getMaxPower() const override;

    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    void setMaxTorque(int32_t MaxTorque) override;
    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    int32_t getMaxTorque() const override;

    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    void setMaxRegenPower(int32_t MaxRegenPower) override;
    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    int32_t getMaxRegenPower() const override;

    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    void setMaxRegenTorque(int32_t MaxRegenTorque) override;
    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    int32_t getMaxRegenTorque() const override;

    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    void setSpeed(int32_t Speed) override;
    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    int32_t getSpeed() const override;

    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    void setTimeInUse(float TimeInUse) override;
    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    float getTimeInUse() const override;

    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    void setTemperature(float Temperature) override;
    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    float getTemperature() const override;

    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    void setCoolantTemperatureDeprecated(float CoolantTemperature_Deprecated) override;
    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    float getCoolantTemperatureDeprecated() const override;

    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    void setPower(int32_t Power) override;
    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    int32_t getPower() const override;

    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    void setTorque(int32_t Torque) override;
    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor implementation. */
    int32_t getTorque() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_ElectricMotor changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Powertrain_ElectricMotor.
    */
    IVehiclePowertrainElectricMotorPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Powertrain_ElectricMotor object which is guarded */
    std::shared_ptr<IVehiclePowertrainElectricMotor> m_impl;
    // Mutex for engineCode property
    mutable std::shared_timed_mutex m_engineCodeMutex;
    // Mutex for maxPower property
    mutable std::shared_timed_mutex m_maxPowerMutex;
    // Mutex for maxTorque property
    mutable std::shared_timed_mutex m_maxTorqueMutex;
    // Mutex for maxRegenPower property
    mutable std::shared_timed_mutex m_maxRegenPowerMutex;
    // Mutex for maxRegenTorque property
    mutable std::shared_timed_mutex m_maxRegenTorqueMutex;
    // Mutex for speed property
    mutable std::shared_timed_mutex m_speedMutex;
    // Mutex for timeInUse property
    mutable std::shared_timed_mutex m_timeInUseMutex;
    // Mutex for temperature property
    mutable std::shared_timed_mutex m_temperatureMutex;
    // Mutex for coolantTemperatureDeprecated property
    mutable std::shared_timed_mutex m_coolantTemperatureDeprecatedMutex;
    // Mutex for power property
    mutable std::shared_timed_mutex m_powerMutex;
    // Mutex for torque property
    mutable std::shared_timed_mutex m_torqueMutex;
};
} // namespace Vehicle
} // namespace Cpp
