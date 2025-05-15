#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehicledriver.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleDriverPublisher.
 * Use this class to store clients of the Vehicle_Driver and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleDriverPublisher : public IVehicleDriverPublisher
{
public:
    /**
    * Implementation of IVehicleDriverPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleDriverSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleDriverPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleDriverSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleDriverPublisher::subscribeToDistractionLevelChanged
    */
    long subscribeToDistractionLevelChanged(VehicleDriverDistractionLevelPropertyCb callback) override;
    /**
    * Implementation of IVehicleDriverPublisher::subscribeToDistractionLevelChanged
    */
    void unsubscribeFromDistractionLevelChanged(long handleId) override;

    /**
    * Implementation of IVehicleDriverPublisher::subscribeToIsEyesOnRoadChanged
    */
    long subscribeToIsEyesOnRoadChanged(VehicleDriverIsEyesOnRoadPropertyCb callback) override;
    /**
    * Implementation of IVehicleDriverPublisher::subscribeToIsEyesOnRoadChanged
    */
    void unsubscribeFromIsEyesOnRoadChanged(long handleId) override;

    /**
    * Implementation of IVehicleDriverPublisher::subscribeToIsHandsOnWheelChanged
    */
    long subscribeToIsHandsOnWheelChanged(VehicleDriverIsHandsOnWheelPropertyCb callback) override;
    /**
    * Implementation of IVehicleDriverPublisher::subscribeToIsHandsOnWheelChanged
    */
    void unsubscribeFromIsHandsOnWheelChanged(long handleId) override;

    /**
    * Implementation of IVehicleDriverPublisher::subscribeToAttentiveProbabilityChanged
    */
    long subscribeToAttentiveProbabilityChanged(VehicleDriverAttentiveProbabilityPropertyCb callback) override;
    /**
    * Implementation of IVehicleDriverPublisher::subscribeToAttentiveProbabilityChanged
    */
    void unsubscribeFromAttentiveProbabilityChanged(long handleId) override;

    /**
    * Implementation of IVehicleDriverPublisher::subscribeToFatigueLevelChanged
    */
    long subscribeToFatigueLevelChanged(VehicleDriverFatigueLevelPropertyCb callback) override;
    /**
    * Implementation of IVehicleDriverPublisher::subscribeToFatigueLevelChanged
    */
    void unsubscribeFromFatigueLevelChanged(long handleId) override;

    /**
    * Implementation of IVehicleDriverPublisher::subscribeToHeartRateChanged
    */
    long subscribeToHeartRateChanged(VehicleDriverHeartRatePropertyCb callback) override;
    /**
    * Implementation of IVehicleDriverPublisher::subscribeToHeartRateChanged
    */
    void unsubscribeFromHeartRateChanged(long handleId) override;

    /**
    * Implementation of IVehicleDriverPublisher::publishDistractionLevelChanged
    */
    void publishDistractionLevelChanged(float DistractionLevel) const override;
    /**
    * Implementation of IVehicleDriverPublisher::publishIsEyesOnRoadChanged
    */
    void publishIsEyesOnRoadChanged(bool IsEyesOnRoad) const override;
    /**
    * Implementation of IVehicleDriverPublisher::publishIsHandsOnWheelChanged
    */
    void publishIsHandsOnWheelChanged(bool IsHandsOnWheel) const override;
    /**
    * Implementation of IVehicleDriverPublisher::publishAttentiveProbabilityChanged
    */
    void publishAttentiveProbabilityChanged(float AttentiveProbability) const override;
    /**
    * Implementation of IVehicleDriverPublisher::publishFatigueLevelChanged
    */
    void publishFatigueLevelChanged(float FatigueLevel) const override;
    /**
    * Implementation of IVehicleDriverPublisher::publishHeartRateChanged
    */
    void publishHeartRateChanged(int32_t HeartRate) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Driver
    std::vector<std::reference_wrapper<IVehicleDriverSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the DistractionLevel change.
    std::atomic<long> m_distractionLevelChangedCallbackNextId {0};
    // Subscribed callbacks for the DistractionLevel change.
    std::map<long, VehicleDriverDistractionLevelPropertyCb> m_distractionLevelCallbacks;
    // Mutex for m_distractionLevelCallbacks
    mutable std::shared_timed_mutex m_distractionLevelCallbacksMutex;
    // Next free unique identifier to subscribe for the IsEyesOnRoad change.
    std::atomic<long> m_isEyesOnRoadChangedCallbackNextId {0};
    // Subscribed callbacks for the IsEyesOnRoad change.
    std::map<long, VehicleDriverIsEyesOnRoadPropertyCb> m_isEyesOnRoadCallbacks;
    // Mutex for m_isEyesOnRoadCallbacks
    mutable std::shared_timed_mutex m_isEyesOnRoadCallbacksMutex;
    // Next free unique identifier to subscribe for the IsHandsOnWheel change.
    std::atomic<long> m_isHandsOnWheelChangedCallbackNextId {0};
    // Subscribed callbacks for the IsHandsOnWheel change.
    std::map<long, VehicleDriverIsHandsOnWheelPropertyCb> m_isHandsOnWheelCallbacks;
    // Mutex for m_isHandsOnWheelCallbacks
    mutable std::shared_timed_mutex m_isHandsOnWheelCallbacksMutex;
    // Next free unique identifier to subscribe for the AttentiveProbability change.
    std::atomic<long> m_attentiveProbabilityChangedCallbackNextId {0};
    // Subscribed callbacks for the AttentiveProbability change.
    std::map<long, VehicleDriverAttentiveProbabilityPropertyCb> m_attentiveProbabilityCallbacks;
    // Mutex for m_attentiveProbabilityCallbacks
    mutable std::shared_timed_mutex m_attentiveProbabilityCallbacksMutex;
    // Next free unique identifier to subscribe for the FatigueLevel change.
    std::atomic<long> m_fatigueLevelChangedCallbackNextId {0};
    // Subscribed callbacks for the FatigueLevel change.
    std::map<long, VehicleDriverFatigueLevelPropertyCb> m_fatigueLevelCallbacks;
    // Mutex for m_fatigueLevelCallbacks
    mutable std::shared_timed_mutex m_fatigueLevelCallbacksMutex;
    // Next free unique identifier to subscribe for the HeartRate change.
    std::atomic<long> m_heartRateChangedCallbackNextId {0};
    // Subscribed callbacks for the HeartRate change.
    std::map<long, VehicleDriverHeartRatePropertyCb> m_heartRateCallbacks;
    // Mutex for m_heartRateCallbacks
    mutable std::shared_timed_mutex m_heartRateCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
