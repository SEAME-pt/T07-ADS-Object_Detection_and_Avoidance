#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclebody.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleBodyPublisher.
 * Use this class to store clients of the Vehicle_Body and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleBodyPublisher : public IVehicleBodyPublisher
{
public:
    /**
    * Implementation of IVehicleBodyPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleBodySubscriber& subscriber) override;
    /**
    * Implementation of IVehicleBodyPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleBodySubscriber& subscriber) override;

    /**
    * Implementation of IVehicleBodyPublisher::subscribeToPowerOptimizeLevelChanged
    */
    long subscribeToPowerOptimizeLevelChanged(VehicleBodyPowerOptimizeLevelPropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyPublisher::subscribeToPowerOptimizeLevelChanged
    */
    void unsubscribeFromPowerOptimizeLevelChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyPublisher::subscribeToIsAutoPowerOptimizeChanged
    */
    long subscribeToIsAutoPowerOptimizeChanged(VehicleBodyIsAutoPowerOptimizePropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyPublisher::subscribeToIsAutoPowerOptimizeChanged
    */
    void unsubscribeFromIsAutoPowerOptimizeChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyPublisher::publishPowerOptimizeLevelChanged
    */
    void publishPowerOptimizeLevelChanged(int32_t PowerOptimizeLevel) const override;
    /**
    * Implementation of IVehicleBodyPublisher::publishIsAutoPowerOptimizeChanged
    */
    void publishIsAutoPowerOptimizeChanged(bool IsAutoPowerOptimize) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Body
    std::vector<std::reference_wrapper<IVehicleBodySubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the PowerOptimizeLevel change.
    std::atomic<long> m_powerOptimizeLevelChangedCallbackNextId {0};
    // Subscribed callbacks for the PowerOptimizeLevel change.
    std::map<long, VehicleBodyPowerOptimizeLevelPropertyCb> m_powerOptimizeLevelCallbacks;
    // Mutex for m_powerOptimizeLevelCallbacks
    mutable std::shared_timed_mutex m_powerOptimizeLevelCallbacksMutex;
    // Next free unique identifier to subscribe for the IsAutoPowerOptimize change.
    std::atomic<long> m_isAutoPowerOptimizeChangedCallbackNextId {0};
    // Subscribed callbacks for the IsAutoPowerOptimize change.
    std::map<long, VehicleBodyIsAutoPowerOptimizePropertyCb> m_isAutoPowerOptimizeCallbacks;
    // Mutex for m_isAutoPowerOptimizeCallbacks
    mutable std::shared_timed_mutex m_isAutoPowerOptimizeCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
