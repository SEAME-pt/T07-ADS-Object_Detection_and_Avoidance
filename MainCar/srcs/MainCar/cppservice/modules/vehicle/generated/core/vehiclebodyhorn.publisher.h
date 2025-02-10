#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclebodyhorn.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleBodyHornPublisher.
 * Use this class to store clients of the Vehicle_Body_Horn and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleBodyHornPublisher : public IVehicleBodyHornPublisher
{
public:
    /**
    * Implementation of IVehicleBodyHornPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleBodyHornSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleBodyHornPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleBodyHornSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleBodyHornPublisher::subscribeToIsActiveChanged
    */
    long subscribeToIsActiveChanged(VehicleBodyHornIsActivePropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyHornPublisher::subscribeToIsActiveChanged
    */
    void unsubscribeFromIsActiveChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyHornPublisher::publishIsActiveChanged
    */
    void publishIsActiveChanged(bool IsActive) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Body_Horn
    std::vector<std::reference_wrapper<IVehicleBodyHornSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the IsActive change.
    std::atomic<long> m_isActiveChangedCallbackNextId {0};
    // Subscribed callbacks for the IsActive change.
    std::map<long, VehicleBodyHornIsActivePropertyCb> m_isActiveCallbacks;
    // Mutex for m_isActiveCallbacks
    mutable std::shared_timed_mutex m_isActiveCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
