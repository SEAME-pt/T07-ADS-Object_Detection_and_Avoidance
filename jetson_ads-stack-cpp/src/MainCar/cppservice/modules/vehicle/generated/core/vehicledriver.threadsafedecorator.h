
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleDriverThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Driver interface.
* @see Vehicle_Driver
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicleDriver> testVehicleDriver = std::make_unique<VehicleDriverThreadSafeDecorator>(std::make_shared<VehicleDriver>());

// Thread safe access
auto distractionLevel = testVehicleDriver->getDistractionLevel();
testVehicleDriver->setDistractionLevel(0.0f);
auto isEyesOnRoad = testVehicleDriver->getIsEyesOnRoad();
testVehicleDriver->setIsEyesOnRoad(false);
auto isHandsOnWheel = testVehicleDriver->getIsHandsOnWheel();
testVehicleDriver->setIsHandsOnWheel(false);
auto attentiveProbability = testVehicleDriver->getAttentiveProbability();
testVehicleDriver->setAttentiveProbability(0.0f);
auto fatigueLevel = testVehicleDriver->getFatigueLevel();
testVehicleDriver->setFatigueLevel(0.0f);
auto heartRate = testVehicleDriver->getHeartRate();
testVehicleDriver->setHeartRate(0);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleDriverThreadSafeDecorator : public IVehicleDriver
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Driver object to make thread safe.
    */
    explicit VehicleDriverThreadSafeDecorator(std::shared_ptr<IVehicleDriver> impl);

    /** Guards and forwards call to Vehicle_Driver implementation. */
    void setDistractionLevel(float DistractionLevel) override;
    /** Guards and forwards call to Vehicle_Driver implementation. */
    float getDistractionLevel() const override;

    /** Guards and forwards call to Vehicle_Driver implementation. */
    void setIsEyesOnRoad(bool IsEyesOnRoad) override;
    /** Guards and forwards call to Vehicle_Driver implementation. */
    bool getIsEyesOnRoad() const override;

    /** Guards and forwards call to Vehicle_Driver implementation. */
    void setIsHandsOnWheel(bool IsHandsOnWheel) override;
    /** Guards and forwards call to Vehicle_Driver implementation. */
    bool getIsHandsOnWheel() const override;

    /** Guards and forwards call to Vehicle_Driver implementation. */
    void setAttentiveProbability(float AttentiveProbability) override;
    /** Guards and forwards call to Vehicle_Driver implementation. */
    float getAttentiveProbability() const override;

    /** Guards and forwards call to Vehicle_Driver implementation. */
    void setFatigueLevel(float FatigueLevel) override;
    /** Guards and forwards call to Vehicle_Driver implementation. */
    float getFatigueLevel() const override;

    /** Guards and forwards call to Vehicle_Driver implementation. */
    void setHeartRate(int32_t HeartRate) override;
    /** Guards and forwards call to Vehicle_Driver implementation. */
    int32_t getHeartRate() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Driver changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Driver.
    */
    IVehicleDriverPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Driver object which is guarded */
    std::shared_ptr<IVehicleDriver> m_impl;
    // Mutex for distractionLevel property
    mutable std::shared_timed_mutex m_distractionLevelMutex;
    // Mutex for isEyesOnRoad property
    mutable std::shared_timed_mutex m_isEyesOnRoadMutex;
    // Mutex for isHandsOnWheel property
    mutable std::shared_timed_mutex m_isHandsOnWheelMutex;
    // Mutex for attentiveProbability property
    mutable std::shared_timed_mutex m_attentiveProbabilityMutex;
    // Mutex for fatigueLevel property
    mutable std::shared_timed_mutex m_fatigueLevelMutex;
    // Mutex for heartRate property
    mutable std::shared_timed_mutex m_heartRateMutex;
};
} // namespace Vehicle
} // namespace Cpp
