#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclebodylightsfogrear.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleBodyLightsFogRearPublisher.
 * Use this class to store clients of the Vehicle_Body_Lights_Fog_Rear and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleBodyLightsFogRearPublisher : public IVehicleBodyLightsFogRearPublisher
{
public:
    /**
    * Implementation of IVehicleBodyLightsFogRearPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleBodyLightsFogRearSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleBodyLightsFogRearPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleBodyLightsFogRearSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleBodyLightsFogRearPublisher::subscribeToIsOnChanged
    */
    long subscribeToIsOnChanged(VehicleBodyLightsFogRearIsOnPropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyLightsFogRearPublisher::subscribeToIsOnChanged
    */
    void unsubscribeFromIsOnChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyLightsFogRearPublisher::subscribeToIsDefectChanged
    */
    long subscribeToIsDefectChanged(VehicleBodyLightsFogRearIsDefectPropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyLightsFogRearPublisher::subscribeToIsDefectChanged
    */
    void unsubscribeFromIsDefectChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyLightsFogRearPublisher::publishIsOnChanged
    */
    void publishIsOnChanged(bool IsOn) const override;
    /**
    * Implementation of IVehicleBodyLightsFogRearPublisher::publishIsDefectChanged
    */
    void publishIsDefectChanged(bool IsDefect) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Body_Lights_Fog_Rear
    std::vector<std::reference_wrapper<IVehicleBodyLightsFogRearSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the IsOn change.
    std::atomic<long> m_isOnChangedCallbackNextId {0};
    // Subscribed callbacks for the IsOn change.
    std::map<long, VehicleBodyLightsFogRearIsOnPropertyCb> m_isOnCallbacks;
    // Mutex for m_isOnCallbacks
    mutable std::shared_timed_mutex m_isOnCallbacksMutex;
    // Next free unique identifier to subscribe for the IsDefect change.
    std::atomic<long> m_isDefectChangedCallbackNextId {0};
    // Subscribed callbacks for the IsDefect change.
    std::map<long, VehicleBodyLightsFogRearIsDefectPropertyCb> m_isDefectCallbacks;
    // Mutex for m_isDefectCallbacks
    mutable std::shared_timed_mutex m_isDefectCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
