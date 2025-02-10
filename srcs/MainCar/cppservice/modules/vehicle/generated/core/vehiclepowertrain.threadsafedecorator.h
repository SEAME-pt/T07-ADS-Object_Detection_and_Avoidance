
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehiclePowertrainThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Powertrain interface.
* @see Vehicle_Powertrain
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehiclePowertrain> testVehiclePowertrain = std::make_unique<VehiclePowertrainThreadSafeDecorator>(std::make_shared<VehiclePowertrain>());

// Thread safe access
auto accumulatedBrakingEnergy = testVehiclePowertrain->getAccumulatedBrakingEnergy();
testVehiclePowertrain->setAccumulatedBrakingEnergy(0.0f);
auto range = testVehiclePowertrain->getRange();
testVehiclePowertrain->setRange(0);
auto timeRemaining = testVehiclePowertrain->getTimeRemaining();
testVehiclePowertrain->setTimeRemaining(0);
auto type = testVehiclePowertrain->getType();
testVehiclePowertrain->setType(Vehicle_Powertrain_Type_ValueEnum::COMBUSTION);
auto powerOptimizeLevel = testVehiclePowertrain->getPowerOptimizeLevel();
testVehiclePowertrain->setPowerOptimizeLevel(0);
auto isAutoPowerOptimize = testVehiclePowertrain->getIsAutoPowerOptimize();
testVehiclePowertrain->setIsAutoPowerOptimize(false);
\endcode
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainThreadSafeDecorator : public IVehiclePowertrain
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Powertrain object to make thread safe.
    */
    explicit VehiclePowertrainThreadSafeDecorator(std::shared_ptr<IVehiclePowertrain> impl);

    /** Guards and forwards call to Vehicle_Powertrain implementation. */
    void setAccumulatedBrakingEnergy(float AccumulatedBrakingEnergy) override;
    /** Guards and forwards call to Vehicle_Powertrain implementation. */
    float getAccumulatedBrakingEnergy() const override;

    /** Guards and forwards call to Vehicle_Powertrain implementation. */
    void setRange(int32_t Range) override;
    /** Guards and forwards call to Vehicle_Powertrain implementation. */
    int32_t getRange() const override;

    /** Guards and forwards call to Vehicle_Powertrain implementation. */
    void setTimeRemaining(int32_t TimeRemaining) override;
    /** Guards and forwards call to Vehicle_Powertrain implementation. */
    int32_t getTimeRemaining() const override;

    /** Guards and forwards call to Vehicle_Powertrain implementation. */
    void setType(Vehicle_Powertrain_Type_ValueEnum Type) override;
    /** Guards and forwards call to Vehicle_Powertrain implementation. */
    Vehicle_Powertrain_Type_ValueEnum getType() const override;

    /** Guards and forwards call to Vehicle_Powertrain implementation. */
    void setPowerOptimizeLevel(int32_t PowerOptimizeLevel) override;
    /** Guards and forwards call to Vehicle_Powertrain implementation. */
    int32_t getPowerOptimizeLevel() const override;

    /** Guards and forwards call to Vehicle_Powertrain implementation. */
    void setIsAutoPowerOptimize(bool IsAutoPowerOptimize) override;
    /** Guards and forwards call to Vehicle_Powertrain implementation. */
    bool getIsAutoPowerOptimize() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Powertrain.
    */
    IVehiclePowertrainPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Powertrain object which is guarded */
    std::shared_ptr<IVehiclePowertrain> m_impl;
    // Mutex for accumulatedBrakingEnergy property
    mutable std::shared_timed_mutex m_accumulatedBrakingEnergyMutex;
    // Mutex for range property
    mutable std::shared_timed_mutex m_rangeMutex;
    // Mutex for timeRemaining property
    mutable std::shared_timed_mutex m_timeRemainingMutex;
    // Mutex for type property
    mutable std::shared_timed_mutex m_typeMutex;
    // Mutex for powerOptimizeLevel property
    mutable std::shared_timed_mutex m_powerOptimizeLevelMutex;
    // Mutex for isAutoPowerOptimize property
    mutable std::shared_timed_mutex m_isAutoPowerOptimizeMutex;
};
} // namespace Vehicle
} // namespace Cpp
