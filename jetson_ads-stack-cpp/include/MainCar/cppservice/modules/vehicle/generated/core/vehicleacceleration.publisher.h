#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehicleacceleration.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleAccelerationPublisher.
 * Use this class to store clients of the Vehicle_Acceleration and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleAccelerationPublisher : public IVehicleAccelerationPublisher
{
public:
    /**
    * Implementation of IVehicleAccelerationPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleAccelerationSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleAccelerationPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleAccelerationSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleAccelerationPublisher::subscribeToLongitudinalChanged
    */
    long subscribeToLongitudinalChanged(VehicleAccelerationLongitudinalPropertyCb callback) override;
    /**
    * Implementation of IVehicleAccelerationPublisher::subscribeToLongitudinalChanged
    */
    void unsubscribeFromLongitudinalChanged(long handleId) override;

    /**
    * Implementation of IVehicleAccelerationPublisher::subscribeToLateralChanged
    */
    long subscribeToLateralChanged(VehicleAccelerationLateralPropertyCb callback) override;
    /**
    * Implementation of IVehicleAccelerationPublisher::subscribeToLateralChanged
    */
    void unsubscribeFromLateralChanged(long handleId) override;

    /**
    * Implementation of IVehicleAccelerationPublisher::subscribeToVerticalChanged
    */
    long subscribeToVerticalChanged(VehicleAccelerationVerticalPropertyCb callback) override;
    /**
    * Implementation of IVehicleAccelerationPublisher::subscribeToVerticalChanged
    */
    void unsubscribeFromVerticalChanged(long handleId) override;

    /**
    * Implementation of IVehicleAccelerationPublisher::publishLongitudinalChanged
    */
    void publishLongitudinalChanged(float Longitudinal) const override;
    /**
    * Implementation of IVehicleAccelerationPublisher::publishLateralChanged
    */
    void publishLateralChanged(float Lateral) const override;
    /**
    * Implementation of IVehicleAccelerationPublisher::publishVerticalChanged
    */
    void publishVerticalChanged(float Vertical) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Acceleration
    std::vector<std::reference_wrapper<IVehicleAccelerationSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Longitudinal change.
    std::atomic<long> m_longitudinalChangedCallbackNextId {0};
    // Subscribed callbacks for the Longitudinal change.
    std::map<long, VehicleAccelerationLongitudinalPropertyCb> m_longitudinalCallbacks;
    // Mutex for m_longitudinalCallbacks
    mutable std::shared_timed_mutex m_longitudinalCallbacksMutex;
    // Next free unique identifier to subscribe for the Lateral change.
    std::atomic<long> m_lateralChangedCallbackNextId {0};
    // Subscribed callbacks for the Lateral change.
    std::map<long, VehicleAccelerationLateralPropertyCb> m_lateralCallbacks;
    // Mutex for m_lateralCallbacks
    mutable std::shared_timed_mutex m_lateralCallbacksMutex;
    // Next free unique identifier to subscribe for the Vertical change.
    std::atomic<long> m_verticalChangedCallbackNextId {0};
    // Subscribed callbacks for the Vertical change.
    std::map<long, VehicleAccelerationVerticalPropertyCb> m_verticalCallbacks;
    // Mutex for m_verticalCallbacks
    mutable std::shared_timed_mutex m_verticalCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
