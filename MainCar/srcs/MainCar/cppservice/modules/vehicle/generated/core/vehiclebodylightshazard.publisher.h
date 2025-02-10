#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclebodylightshazard.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleBodyLightsHazardPublisher.
 * Use this class to store clients of the Vehicle_Body_Lights_Hazard and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleBodyLightsHazardPublisher : public IVehicleBodyLightsHazardPublisher
{
public:
    /**
    * Implementation of IVehicleBodyLightsHazardPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleBodyLightsHazardSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleBodyLightsHazardPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleBodyLightsHazardSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleBodyLightsHazardPublisher::subscribeToIsSignalingChanged
    */
    long subscribeToIsSignalingChanged(VehicleBodyLightsHazardIsSignalingPropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyLightsHazardPublisher::subscribeToIsSignalingChanged
    */
    void unsubscribeFromIsSignalingChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyLightsHazardPublisher::subscribeToIsDefectChanged
    */
    long subscribeToIsDefectChanged(VehicleBodyLightsHazardIsDefectPropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyLightsHazardPublisher::subscribeToIsDefectChanged
    */
    void unsubscribeFromIsDefectChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyLightsHazardPublisher::publishIsSignalingChanged
    */
    void publishIsSignalingChanged(bool IsSignaling) const override;
    /**
    * Implementation of IVehicleBodyLightsHazardPublisher::publishIsDefectChanged
    */
    void publishIsDefectChanged(bool IsDefect) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Body_Lights_Hazard
    std::vector<std::reference_wrapper<IVehicleBodyLightsHazardSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the IsSignaling change.
    std::atomic<long> m_isSignalingChangedCallbackNextId {0};
    // Subscribed callbacks for the IsSignaling change.
    std::map<long, VehicleBodyLightsHazardIsSignalingPropertyCb> m_isSignalingCallbacks;
    // Mutex for m_isSignalingCallbacks
    mutable std::shared_timed_mutex m_isSignalingCallbacksMutex;
    // Next free unique identifier to subscribe for the IsDefect change.
    std::atomic<long> m_isDefectChangedCallbackNextId {0};
    // Subscribed callbacks for the IsDefect change.
    std::map<long, VehicleBodyLightsHazardIsDefectPropertyCb> m_isDefectCallbacks;
    // Mutex for m_isDefectCallbacks
    mutable std::shared_timed_mutex m_isDefectCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
