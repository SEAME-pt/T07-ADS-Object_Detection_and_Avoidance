
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehiclePowertrainElectricMotorEngineCoolantThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Powertrain_ElectricMotor_EngineCoolant interface.
* @see Vehicle_Powertrain_ElectricMotor_EngineCoolant
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehiclePowertrainElectricMotorEngineCoolant> testVehiclePowertrainElectricMotorEngineCoolant = std::make_unique<VehiclePowertrainElectricMotorEngineCoolantThreadSafeDecorator>(std::make_shared<VehiclePowertrainElectricMotorEngineCoolant>());

// Thread safe access
auto capacity = testVehiclePowertrainElectricMotorEngineCoolant->getCapacity();
testVehiclePowertrainElectricMotorEngineCoolant->setCapacity(0.0f);
auto temperature = testVehiclePowertrainElectricMotorEngineCoolant->getTemperature();
testVehiclePowertrainElectricMotorEngineCoolant->setTemperature(0.0f);
auto level = testVehiclePowertrainElectricMotorEngineCoolant->getLevel();
testVehiclePowertrainElectricMotorEngineCoolant->setLevel(Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum::CRITICALLY_LOW);
auto lifeRemaining = testVehiclePowertrainElectricMotorEngineCoolant->getLifeRemaining();
testVehiclePowertrainElectricMotorEngineCoolant->setLifeRemaining(0);
\endcode
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainElectricMotorEngineCoolantThreadSafeDecorator : public IVehiclePowertrainElectricMotorEngineCoolant
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Powertrain_ElectricMotor_EngineCoolant object to make thread safe.
    */
    explicit VehiclePowertrainElectricMotorEngineCoolantThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainElectricMotorEngineCoolant> impl);

    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor_EngineCoolant implementation. */
    void setCapacity(float Capacity) override;
    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor_EngineCoolant implementation. */
    float getCapacity() const override;

    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor_EngineCoolant implementation. */
    void setTemperature(float Temperature) override;
    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor_EngineCoolant implementation. */
    float getTemperature() const override;

    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor_EngineCoolant implementation. */
    void setLevel(Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum Level) override;
    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor_EngineCoolant implementation. */
    Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum getLevel() const override;

    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor_EngineCoolant implementation. */
    void setLifeRemaining(int32_t LifeRemaining) override;
    /** Guards and forwards call to Vehicle_Powertrain_ElectricMotor_EngineCoolant implementation. */
    int32_t getLifeRemaining() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_ElectricMotor_EngineCoolant changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Powertrain_ElectricMotor_EngineCoolant.
    */
    IVehiclePowertrainElectricMotorEngineCoolantPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Powertrain_ElectricMotor_EngineCoolant object which is guarded */
    std::shared_ptr<IVehiclePowertrainElectricMotorEngineCoolant> m_impl;
    // Mutex for capacity property
    mutable std::shared_timed_mutex m_capacityMutex;
    // Mutex for temperature property
    mutable std::shared_timed_mutex m_temperatureMutex;
    // Mutex for level property
    mutable std::shared_timed_mutex m_levelMutex;
    // Mutex for lifeRemaining property
    mutable std::shared_timed_mutex m_lifeRemainingMutex;
};
} // namespace Vehicle
} // namespace Cpp
