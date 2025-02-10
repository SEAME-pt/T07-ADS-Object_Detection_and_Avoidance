#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehicle.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePublisher.
 * Use this class to store clients of the Vehicle and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePublisher : public IVehiclePublisher
{
public:
    /**
    * Implementation of IVehiclePublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePublisher::subscribeToLowVoltageSystemStateChanged
    */
    long subscribeToLowVoltageSystemStateChanged(VehicleLowVoltageSystemStatePropertyCb callback) override;
    /**
    * Implementation of IVehiclePublisher::subscribeToLowVoltageSystemStateChanged
    */
    void unsubscribeFromLowVoltageSystemStateChanged(long handleId) override;

    /**
    * Implementation of IVehiclePublisher::subscribeToSpeedChanged
    */
    long subscribeToSpeedChanged(VehicleSpeedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePublisher::subscribeToSpeedChanged
    */
    void unsubscribeFromSpeedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePublisher::subscribeToTraveledDistanceChanged
    */
    long subscribeToTraveledDistanceChanged(VehicleTraveledDistancePropertyCb callback) override;
    /**
    * Implementation of IVehiclePublisher::subscribeToTraveledDistanceChanged
    */
    void unsubscribeFromTraveledDistanceChanged(long handleId) override;

    /**
    * Implementation of IVehiclePublisher::subscribeToTraveledDistanceSinceStartChanged
    */
    long subscribeToTraveledDistanceSinceStartChanged(VehicleTraveledDistanceSinceStartPropertyCb callback) override;
    /**
    * Implementation of IVehiclePublisher::subscribeToTraveledDistanceSinceStartChanged
    */
    void unsubscribeFromTraveledDistanceSinceStartChanged(long handleId) override;

    /**
    * Implementation of IVehiclePublisher::subscribeToStartTimeChanged
    */
    long subscribeToStartTimeChanged(VehicleStartTimePropertyCb callback) override;
    /**
    * Implementation of IVehiclePublisher::subscribeToStartTimeChanged
    */
    void unsubscribeFromStartTimeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePublisher::subscribeToTripDurationChanged
    */
    long subscribeToTripDurationChanged(VehicleTripDurationPropertyCb callback) override;
    /**
    * Implementation of IVehiclePublisher::subscribeToTripDurationChanged
    */
    void unsubscribeFromTripDurationChanged(long handleId) override;

    /**
    * Implementation of IVehiclePublisher::subscribeToTripMeterReadingChanged
    */
    long subscribeToTripMeterReadingChanged(VehicleTripMeterReadingPropertyCb callback) override;
    /**
    * Implementation of IVehiclePublisher::subscribeToTripMeterReadingChanged
    */
    void unsubscribeFromTripMeterReadingChanged(long handleId) override;

    /**
    * Implementation of IVehiclePublisher::subscribeToIsBrokenDownChanged
    */
    long subscribeToIsBrokenDownChanged(VehicleIsBrokenDownPropertyCb callback) override;
    /**
    * Implementation of IVehiclePublisher::subscribeToIsBrokenDownChanged
    */
    void unsubscribeFromIsBrokenDownChanged(long handleId) override;

    /**
    * Implementation of IVehiclePublisher::subscribeToIsMovingChanged
    */
    long subscribeToIsMovingChanged(VehicleIsMovingPropertyCb callback) override;
    /**
    * Implementation of IVehiclePublisher::subscribeToIsMovingChanged
    */
    void unsubscribeFromIsMovingChanged(long handleId) override;

    /**
    * Implementation of IVehiclePublisher::subscribeToAverageSpeedChanged
    */
    long subscribeToAverageSpeedChanged(VehicleAverageSpeedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePublisher::subscribeToAverageSpeedChanged
    */
    void unsubscribeFromAverageSpeedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePublisher::subscribeToPowerOptimizeLevelChanged
    */
    long subscribeToPowerOptimizeLevelChanged(VehiclePowerOptimizeLevelPropertyCb callback) override;
    /**
    * Implementation of IVehiclePublisher::subscribeToPowerOptimizeLevelChanged
    */
    void unsubscribeFromPowerOptimizeLevelChanged(long handleId) override;

    /**
    * Implementation of IVehiclePublisher::subscribeToIsAutoPowerOptimizeChanged
    */
    long subscribeToIsAutoPowerOptimizeChanged(VehicleIsAutoPowerOptimizePropertyCb callback) override;
    /**
    * Implementation of IVehiclePublisher::subscribeToIsAutoPowerOptimizeChanged
    */
    void unsubscribeFromIsAutoPowerOptimizeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePublisher::publishLowVoltageSystemStateChanged
    */
    void publishLowVoltageSystemStateChanged(Vehicle_LowVoltageSystemState_ValueEnum LowVoltageSystemState) const override;
    /**
    * Implementation of IVehiclePublisher::publishSpeedChanged
    */
    void publishSpeedChanged(float Speed) const override;
    /**
    * Implementation of IVehiclePublisher::publishTraveledDistanceChanged
    */
    void publishTraveledDistanceChanged(float TraveledDistance) const override;
    /**
    * Implementation of IVehiclePublisher::publishTraveledDistanceSinceStartChanged
    */
    void publishTraveledDistanceSinceStartChanged(float TraveledDistanceSinceStart) const override;
    /**
    * Implementation of IVehiclePublisher::publishStartTimeChanged
    */
    void publishStartTimeChanged(const std::string& StartTime) const override;
    /**
    * Implementation of IVehiclePublisher::publishTripDurationChanged
    */
    void publishTripDurationChanged(float TripDuration) const override;
    /**
    * Implementation of IVehiclePublisher::publishTripMeterReadingChanged
    */
    void publishTripMeterReadingChanged(float TripMeterReading) const override;
    /**
    * Implementation of IVehiclePublisher::publishIsBrokenDownChanged
    */
    void publishIsBrokenDownChanged(bool IsBrokenDown) const override;
    /**
    * Implementation of IVehiclePublisher::publishIsMovingChanged
    */
    void publishIsMovingChanged(bool IsMoving) const override;
    /**
    * Implementation of IVehiclePublisher::publishAverageSpeedChanged
    */
    void publishAverageSpeedChanged(float AverageSpeed) const override;
    /**
    * Implementation of IVehiclePublisher::publishPowerOptimizeLevelChanged
    */
    void publishPowerOptimizeLevelChanged(int32_t PowerOptimizeLevel) const override;
    /**
    * Implementation of IVehiclePublisher::publishIsAutoPowerOptimizeChanged
    */
    void publishIsAutoPowerOptimizeChanged(bool IsAutoPowerOptimize) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle
    std::vector<std::reference_wrapper<IVehicleSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the LowVoltageSystemState change.
    std::atomic<long> m_lowVoltageSystemStateChangedCallbackNextId {0};
    // Subscribed callbacks for the LowVoltageSystemState change.
    std::map<long, VehicleLowVoltageSystemStatePropertyCb> m_lowVoltageSystemStateCallbacks;
    // Mutex for m_lowVoltageSystemStateCallbacks
    mutable std::shared_timed_mutex m_lowVoltageSystemStateCallbacksMutex;
    // Next free unique identifier to subscribe for the Speed change.
    std::atomic<long> m_speedChangedCallbackNextId {0};
    // Subscribed callbacks for the Speed change.
    std::map<long, VehicleSpeedPropertyCb> m_speedCallbacks;
    // Mutex for m_speedCallbacks
    mutable std::shared_timed_mutex m_speedCallbacksMutex;
    // Next free unique identifier to subscribe for the TraveledDistance change.
    std::atomic<long> m_traveledDistanceChangedCallbackNextId {0};
    // Subscribed callbacks for the TraveledDistance change.
    std::map<long, VehicleTraveledDistancePropertyCb> m_traveledDistanceCallbacks;
    // Mutex for m_traveledDistanceCallbacks
    mutable std::shared_timed_mutex m_traveledDistanceCallbacksMutex;
    // Next free unique identifier to subscribe for the TraveledDistanceSinceStart change.
    std::atomic<long> m_traveledDistanceSinceStartChangedCallbackNextId {0};
    // Subscribed callbacks for the TraveledDistanceSinceStart change.
    std::map<long, VehicleTraveledDistanceSinceStartPropertyCb> m_traveledDistanceSinceStartCallbacks;
    // Mutex for m_traveledDistanceSinceStartCallbacks
    mutable std::shared_timed_mutex m_traveledDistanceSinceStartCallbacksMutex;
    // Next free unique identifier to subscribe for the StartTime change.
    std::atomic<long> m_startTimeChangedCallbackNextId {0};
    // Subscribed callbacks for the StartTime change.
    std::map<long, VehicleStartTimePropertyCb> m_startTimeCallbacks;
    // Mutex for m_startTimeCallbacks
    mutable std::shared_timed_mutex m_startTimeCallbacksMutex;
    // Next free unique identifier to subscribe for the TripDuration change.
    std::atomic<long> m_tripDurationChangedCallbackNextId {0};
    // Subscribed callbacks for the TripDuration change.
    std::map<long, VehicleTripDurationPropertyCb> m_tripDurationCallbacks;
    // Mutex for m_tripDurationCallbacks
    mutable std::shared_timed_mutex m_tripDurationCallbacksMutex;
    // Next free unique identifier to subscribe for the TripMeterReading change.
    std::atomic<long> m_tripMeterReadingChangedCallbackNextId {0};
    // Subscribed callbacks for the TripMeterReading change.
    std::map<long, VehicleTripMeterReadingPropertyCb> m_tripMeterReadingCallbacks;
    // Mutex for m_tripMeterReadingCallbacks
    mutable std::shared_timed_mutex m_tripMeterReadingCallbacksMutex;
    // Next free unique identifier to subscribe for the IsBrokenDown change.
    std::atomic<long> m_isBrokenDownChangedCallbackNextId {0};
    // Subscribed callbacks for the IsBrokenDown change.
    std::map<long, VehicleIsBrokenDownPropertyCb> m_isBrokenDownCallbacks;
    // Mutex for m_isBrokenDownCallbacks
    mutable std::shared_timed_mutex m_isBrokenDownCallbacksMutex;
    // Next free unique identifier to subscribe for the IsMoving change.
    std::atomic<long> m_isMovingChangedCallbackNextId {0};
    // Subscribed callbacks for the IsMoving change.
    std::map<long, VehicleIsMovingPropertyCb> m_isMovingCallbacks;
    // Mutex for m_isMovingCallbacks
    mutable std::shared_timed_mutex m_isMovingCallbacksMutex;
    // Next free unique identifier to subscribe for the AverageSpeed change.
    std::atomic<long> m_averageSpeedChangedCallbackNextId {0};
    // Subscribed callbacks for the AverageSpeed change.
    std::map<long, VehicleAverageSpeedPropertyCb> m_averageSpeedCallbacks;
    // Mutex for m_averageSpeedCallbacks
    mutable std::shared_timed_mutex m_averageSpeedCallbacksMutex;
    // Next free unique identifier to subscribe for the PowerOptimizeLevel change.
    std::atomic<long> m_powerOptimizeLevelChangedCallbackNextId {0};
    // Subscribed callbacks for the PowerOptimizeLevel change.
    std::map<long, VehiclePowerOptimizeLevelPropertyCb> m_powerOptimizeLevelCallbacks;
    // Mutex for m_powerOptimizeLevelCallbacks
    mutable std::shared_timed_mutex m_powerOptimizeLevelCallbacksMutex;
    // Next free unique identifier to subscribe for the IsAutoPowerOptimize change.
    std::atomic<long> m_isAutoPowerOptimizeChangedCallbackNextId {0};
    // Subscribed callbacks for the IsAutoPowerOptimize change.
    std::map<long, VehicleIsAutoPowerOptimizePropertyCb> m_isAutoPowerOptimizeCallbacks;
    // Mutex for m_isAutoPowerOptimizeCallbacks
    mutable std::shared_timed_mutex m_isAutoPowerOptimizeCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
