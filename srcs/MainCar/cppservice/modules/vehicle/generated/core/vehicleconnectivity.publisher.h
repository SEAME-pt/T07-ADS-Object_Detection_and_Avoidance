#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehicleconnectivity.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleConnectivityPublisher.
 * Use this class to store clients of the Vehicle_Connectivity and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleConnectivityPublisher : public IVehicleConnectivityPublisher
{
public:
    /**
    * Implementation of IVehicleConnectivityPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleConnectivitySubscriber& subscriber) override;
    /**
    * Implementation of IVehicleConnectivityPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleConnectivitySubscriber& subscriber) override;

    /**
    * Implementation of IVehicleConnectivityPublisher::subscribeToIsConnectivityAvailableChanged
    */
    long subscribeToIsConnectivityAvailableChanged(VehicleConnectivityIsConnectivityAvailablePropertyCb callback) override;
    /**
    * Implementation of IVehicleConnectivityPublisher::subscribeToIsConnectivityAvailableChanged
    */
    void unsubscribeFromIsConnectivityAvailableChanged(long handleId) override;

    /**
    * Implementation of IVehicleConnectivityPublisher::publishIsConnectivityAvailableChanged
    */
    void publishIsConnectivityAvailableChanged(bool IsConnectivityAvailable) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Connectivity
    std::vector<std::reference_wrapper<IVehicleConnectivitySubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the IsConnectivityAvailable change.
    std::atomic<long> m_isConnectivityAvailableChangedCallbackNextId {0};
    // Subscribed callbacks for the IsConnectivityAvailable change.
    std::map<long, VehicleConnectivityIsConnectivityAvailablePropertyCb> m_isConnectivityAvailableCallbacks;
    // Mutex for m_isConnectivityAvailableCallbacks
    mutable std::shared_timed_mutex m_isConnectivityAvailableCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
