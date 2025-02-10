#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclebodylightsparking.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleBodyLightsParkingPublisher.
 * Use this class to store clients of the Vehicle_Body_Lights_Parking and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleBodyLightsParkingPublisher : public IVehicleBodyLightsParkingPublisher
{
public:
    /**
    * Implementation of IVehicleBodyLightsParkingPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleBodyLightsParkingSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleBodyLightsParkingPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleBodyLightsParkingSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleBodyLightsParkingPublisher::subscribeToIsOnChanged
    */
    long subscribeToIsOnChanged(VehicleBodyLightsParkingIsOnPropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyLightsParkingPublisher::subscribeToIsOnChanged
    */
    void unsubscribeFromIsOnChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyLightsParkingPublisher::subscribeToIsDefectChanged
    */
    long subscribeToIsDefectChanged(VehicleBodyLightsParkingIsDefectPropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyLightsParkingPublisher::subscribeToIsDefectChanged
    */
    void unsubscribeFromIsDefectChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyLightsParkingPublisher::publishIsOnChanged
    */
    void publishIsOnChanged(bool IsOn) const override;
    /**
    * Implementation of IVehicleBodyLightsParkingPublisher::publishIsDefectChanged
    */
    void publishIsDefectChanged(bool IsDefect) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Body_Lights_Parking
    std::vector<std::reference_wrapper<IVehicleBodyLightsParkingSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the IsOn change.
    std::atomic<long> m_isOnChangedCallbackNextId {0};
    // Subscribed callbacks for the IsOn change.
    std::map<long, VehicleBodyLightsParkingIsOnPropertyCb> m_isOnCallbacks;
    // Mutex for m_isOnCallbacks
    mutable std::shared_timed_mutex m_isOnCallbacksMutex;
    // Next free unique identifier to subscribe for the IsDefect change.
    std::atomic<long> m_isDefectChangedCallbackNextId {0};
    // Subscribed callbacks for the IsDefect change.
    std::map<long, VehicleBodyLightsParkingIsDefectPropertyCb> m_isDefectCallbacks;
    // Mutex for m_isDefectCallbacks
    mutable std::shared_timed_mutex m_isDefectCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
