#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehicleangularvelocity.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleAngularVelocityPublisher.
 * Use this class to store clients of the Vehicle_AngularVelocity and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleAngularVelocityPublisher : public IVehicleAngularVelocityPublisher
{
public:
    /**
    * Implementation of IVehicleAngularVelocityPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleAngularVelocitySubscriber& subscriber) override;
    /**
    * Implementation of IVehicleAngularVelocityPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleAngularVelocitySubscriber& subscriber) override;

    /**
    * Implementation of IVehicleAngularVelocityPublisher::subscribeToRollChanged
    */
    long subscribeToRollChanged(VehicleAngularVelocityRollPropertyCb callback) override;
    /**
    * Implementation of IVehicleAngularVelocityPublisher::subscribeToRollChanged
    */
    void unsubscribeFromRollChanged(long handleId) override;

    /**
    * Implementation of IVehicleAngularVelocityPublisher::subscribeToPitchChanged
    */
    long subscribeToPitchChanged(VehicleAngularVelocityPitchPropertyCb callback) override;
    /**
    * Implementation of IVehicleAngularVelocityPublisher::subscribeToPitchChanged
    */
    void unsubscribeFromPitchChanged(long handleId) override;

    /**
    * Implementation of IVehicleAngularVelocityPublisher::subscribeToYawChanged
    */
    long subscribeToYawChanged(VehicleAngularVelocityYawPropertyCb callback) override;
    /**
    * Implementation of IVehicleAngularVelocityPublisher::subscribeToYawChanged
    */
    void unsubscribeFromYawChanged(long handleId) override;

    /**
    * Implementation of IVehicleAngularVelocityPublisher::publishRollChanged
    */
    void publishRollChanged(float Roll) const override;
    /**
    * Implementation of IVehicleAngularVelocityPublisher::publishPitchChanged
    */
    void publishPitchChanged(float Pitch) const override;
    /**
    * Implementation of IVehicleAngularVelocityPublisher::publishYawChanged
    */
    void publishYawChanged(float Yaw) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_AngularVelocity
    std::vector<std::reference_wrapper<IVehicleAngularVelocitySubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Roll change.
    std::atomic<long> m_rollChangedCallbackNextId {0};
    // Subscribed callbacks for the Roll change.
    std::map<long, VehicleAngularVelocityRollPropertyCb> m_rollCallbacks;
    // Mutex for m_rollCallbacks
    mutable std::shared_timed_mutex m_rollCallbacksMutex;
    // Next free unique identifier to subscribe for the Pitch change.
    std::atomic<long> m_pitchChangedCallbackNextId {0};
    // Subscribed callbacks for the Pitch change.
    std::map<long, VehicleAngularVelocityPitchPropertyCb> m_pitchCallbacks;
    // Mutex for m_pitchCallbacks
    mutable std::shared_timed_mutex m_pitchCallbacksMutex;
    // Next free unique identifier to subscribe for the Yaw change.
    std::atomic<long> m_yawChangedCallbackNextId {0};
    // Subscribed callbacks for the Yaw change.
    std::map<long, VehicleAngularVelocityYawPropertyCb> m_yawCallbacks;
    // Mutex for m_yawCallbacks
    mutable std::shared_timed_mutex m_yawCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
