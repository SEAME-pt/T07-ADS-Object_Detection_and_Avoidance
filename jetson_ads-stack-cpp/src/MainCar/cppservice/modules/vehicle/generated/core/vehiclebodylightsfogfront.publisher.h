#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclebodylightsfogfront.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleBodyLightsFogFrontPublisher.
 * Use this class to store clients of the Vehicle_Body_Lights_Fog_Front and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleBodyLightsFogFrontPublisher : public IVehicleBodyLightsFogFrontPublisher
{
public:
    /**
    * Implementation of IVehicleBodyLightsFogFrontPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleBodyLightsFogFrontSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleBodyLightsFogFrontPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleBodyLightsFogFrontSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleBodyLightsFogFrontPublisher::subscribeToIsOnChanged
    */
    long subscribeToIsOnChanged(VehicleBodyLightsFogFrontIsOnPropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyLightsFogFrontPublisher::subscribeToIsOnChanged
    */
    void unsubscribeFromIsOnChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyLightsFogFrontPublisher::subscribeToIsDefectChanged
    */
    long subscribeToIsDefectChanged(VehicleBodyLightsFogFrontIsDefectPropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyLightsFogFrontPublisher::subscribeToIsDefectChanged
    */
    void unsubscribeFromIsDefectChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyLightsFogFrontPublisher::publishIsOnChanged
    */
    void publishIsOnChanged(bool IsOn) const override;
    /**
    * Implementation of IVehicleBodyLightsFogFrontPublisher::publishIsDefectChanged
    */
    void publishIsDefectChanged(bool IsDefect) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Body_Lights_Fog_Front
    std::vector<std::reference_wrapper<IVehicleBodyLightsFogFrontSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the IsOn change.
    std::atomic<long> m_isOnChangedCallbackNextId {0};
    // Subscribed callbacks for the IsOn change.
    std::map<long, VehicleBodyLightsFogFrontIsOnPropertyCb> m_isOnCallbacks;
    // Mutex for m_isOnCallbacks
    mutable std::shared_timed_mutex m_isOnCallbacksMutex;
    // Next free unique identifier to subscribe for the IsDefect change.
    std::atomic<long> m_isDefectChangedCallbackNextId {0};
    // Subscribed callbacks for the IsDefect change.
    std::map<long, VehicleBodyLightsFogFrontIsDefectPropertyCb> m_isDefectCallbacks;
    // Mutex for m_isDefectCallbacks
    mutable std::shared_timed_mutex m_isDefectCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
