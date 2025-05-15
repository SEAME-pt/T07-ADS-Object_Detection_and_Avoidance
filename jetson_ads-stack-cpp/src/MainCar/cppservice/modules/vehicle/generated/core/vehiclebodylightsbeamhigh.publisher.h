#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclebodylightsbeamhigh.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleBodyLightsBeamHighPublisher.
 * Use this class to store clients of the Vehicle_Body_Lights_Beam_High and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleBodyLightsBeamHighPublisher : public IVehicleBodyLightsBeamHighPublisher
{
public:
    /**
    * Implementation of IVehicleBodyLightsBeamHighPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleBodyLightsBeamHighSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleBodyLightsBeamHighPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleBodyLightsBeamHighSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleBodyLightsBeamHighPublisher::subscribeToIsOnChanged
    */
    long subscribeToIsOnChanged(VehicleBodyLightsBeamHighIsOnPropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyLightsBeamHighPublisher::subscribeToIsOnChanged
    */
    void unsubscribeFromIsOnChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyLightsBeamHighPublisher::subscribeToIsDefectChanged
    */
    long subscribeToIsDefectChanged(VehicleBodyLightsBeamHighIsDefectPropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyLightsBeamHighPublisher::subscribeToIsDefectChanged
    */
    void unsubscribeFromIsDefectChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyLightsBeamHighPublisher::publishIsOnChanged
    */
    void publishIsOnChanged(bool IsOn) const override;
    /**
    * Implementation of IVehicleBodyLightsBeamHighPublisher::publishIsDefectChanged
    */
    void publishIsDefectChanged(bool IsDefect) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Body_Lights_Beam_High
    std::vector<std::reference_wrapper<IVehicleBodyLightsBeamHighSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the IsOn change.
    std::atomic<long> m_isOnChangedCallbackNextId {0};
    // Subscribed callbacks for the IsOn change.
    std::map<long, VehicleBodyLightsBeamHighIsOnPropertyCb> m_isOnCallbacks;
    // Mutex for m_isOnCallbacks
    mutable std::shared_timed_mutex m_isOnCallbacksMutex;
    // Next free unique identifier to subscribe for the IsDefect change.
    std::atomic<long> m_isDefectChangedCallbackNextId {0};
    // Subscribed callbacks for the IsDefect change.
    std::map<long, VehicleBodyLightsBeamHighIsDefectPropertyCb> m_isDefectCallbacks;
    // Mutex for m_isDefectCallbacks
    mutable std::shared_timed_mutex m_isDefectCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
