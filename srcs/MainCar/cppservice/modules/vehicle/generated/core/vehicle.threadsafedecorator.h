
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehicleThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle interface.
* @see Vehicle
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehicle> testVehicle = std::make_unique<VehicleThreadSafeDecorator>(std::make_shared<Vehicle>());

// Thread safe access
auto lowVoltageSystemState = testVehicle->getLowVoltageSystemState();
testVehicle->setLowVoltageSystemState(Vehicle_LowVoltageSystemState_ValueEnum::UNDEFINED);
auto speed = testVehicle->getSpeed();
testVehicle->setSpeed(0.0f);
auto traveledDistance = testVehicle->getTraveledDistance();
testVehicle->setTraveledDistance(0.0f);
auto traveledDistanceSinceStart = testVehicle->getTraveledDistanceSinceStart();
testVehicle->setTraveledDistanceSinceStart(0.0f);
auto startTime = testVehicle->getStartTime();
testVehicle->setStartTime(std::string());
auto tripDuration = testVehicle->getTripDuration();
testVehicle->setTripDuration(0.0f);
auto tripMeterReading = testVehicle->getTripMeterReading();
testVehicle->setTripMeterReading(0.0f);
auto isBrokenDown = testVehicle->getIsBrokenDown();
testVehicle->setIsBrokenDown(false);
auto isMoving = testVehicle->getIsMoving();
testVehicle->setIsMoving(false);
auto averageSpeed = testVehicle->getAverageSpeed();
testVehicle->setAverageSpeed(0.0f);
auto powerOptimizeLevel = testVehicle->getPowerOptimizeLevel();
testVehicle->setPowerOptimizeLevel(0);
auto isAutoPowerOptimize = testVehicle->getIsAutoPowerOptimize();
testVehicle->setIsAutoPowerOptimize(false);
\endcode
*/
class CPP_VEHICLE_EXPORT VehicleThreadSafeDecorator : public IVehicle
{
public:
    /** 
    * ctor
    * @param impl The Vehicle object to make thread safe.
    */
    explicit VehicleThreadSafeDecorator(std::shared_ptr<IVehicle> impl);

    /** Guards and forwards call to Vehicle implementation. */
    void setLowVoltageSystemState(Vehicle_LowVoltageSystemState_ValueEnum LowVoltageSystemState) override;
    /** Guards and forwards call to Vehicle implementation. */
    Vehicle_LowVoltageSystemState_ValueEnum getLowVoltageSystemState() const override;

    /** Guards and forwards call to Vehicle implementation. */
    void setSpeed(float Speed) override;
    /** Guards and forwards call to Vehicle implementation. */
    float getSpeed() const override;

    /** Guards and forwards call to Vehicle implementation. */
    void setTraveledDistance(float TraveledDistance) override;
    /** Guards and forwards call to Vehicle implementation. */
    float getTraveledDistance() const override;

    /** Guards and forwards call to Vehicle implementation. */
    void setTraveledDistanceSinceStart(float TraveledDistanceSinceStart) override;
    /** Guards and forwards call to Vehicle implementation. */
    float getTraveledDistanceSinceStart() const override;

    /** Guards and forwards call to Vehicle implementation. */
    void setStartTime(const std::string& StartTime) override;
    /** Guards and forwards call to Vehicle implementation. */
    const std::string& getStartTime() const override;

    /** Guards and forwards call to Vehicle implementation. */
    void setTripDuration(float TripDuration) override;
    /** Guards and forwards call to Vehicle implementation. */
    float getTripDuration() const override;

    /** Guards and forwards call to Vehicle implementation. */
    void setTripMeterReading(float TripMeterReading) override;
    /** Guards and forwards call to Vehicle implementation. */
    float getTripMeterReading() const override;

    /** Guards and forwards call to Vehicle implementation. */
    void setIsBrokenDown(bool IsBrokenDown) override;
    /** Guards and forwards call to Vehicle implementation. */
    bool getIsBrokenDown() const override;

    /** Guards and forwards call to Vehicle implementation. */
    void setIsMoving(bool IsMoving) override;
    /** Guards and forwards call to Vehicle implementation. */
    bool getIsMoving() const override;

    /** Guards and forwards call to Vehicle implementation. */
    void setAverageSpeed(float AverageSpeed) override;
    /** Guards and forwards call to Vehicle implementation. */
    float getAverageSpeed() const override;

    /** Guards and forwards call to Vehicle implementation. */
    void setPowerOptimizeLevel(int32_t PowerOptimizeLevel) override;
    /** Guards and forwards call to Vehicle implementation. */
    int32_t getPowerOptimizeLevel() const override;

    /** Guards and forwards call to Vehicle implementation. */
    void setIsAutoPowerOptimize(bool IsAutoPowerOptimize) override;
    /** Guards and forwards call to Vehicle implementation. */
    bool getIsAutoPowerOptimize() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle.
    */
    IVehiclePublisher& _getPublisher() const override;
private:
    /** The Vehicle object which is guarded */
    std::shared_ptr<IVehicle> m_impl;
    // Mutex for lowVoltageSystemState property
    mutable std::shared_timed_mutex m_lowVoltageSystemStateMutex;
    // Mutex for speed property
    mutable std::shared_timed_mutex m_speedMutex;
    // Mutex for traveledDistance property
    mutable std::shared_timed_mutex m_traveledDistanceMutex;
    // Mutex for traveledDistanceSinceStart property
    mutable std::shared_timed_mutex m_traveledDistanceSinceStartMutex;
    // Mutex for startTime property
    mutable std::shared_timed_mutex m_startTimeMutex;
    // Mutex for tripDuration property
    mutable std::shared_timed_mutex m_tripDurationMutex;
    // Mutex for tripMeterReading property
    mutable std::shared_timed_mutex m_tripMeterReadingMutex;
    // Mutex for isBrokenDown property
    mutable std::shared_timed_mutex m_isBrokenDownMutex;
    // Mutex for isMoving property
    mutable std::shared_timed_mutex m_isMovingMutex;
    // Mutex for averageSpeed property
    mutable std::shared_timed_mutex m_averageSpeedMutex;
    // Mutex for powerOptimizeLevel property
    mutable std::shared_timed_mutex m_powerOptimizeLevelMutex;
    // Mutex for isAutoPowerOptimize property
    mutable std::shared_timed_mutex m_isAutoPowerOptimizeMutex;
};
} // namespace Vehicle
} // namespace Cpp
