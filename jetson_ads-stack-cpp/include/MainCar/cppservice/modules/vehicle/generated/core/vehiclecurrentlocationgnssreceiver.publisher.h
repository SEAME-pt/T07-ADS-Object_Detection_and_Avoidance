#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclecurrentlocationgnssreceiver.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleCurrentLocationGnssReceiverPublisher.
 * Use this class to store clients of the Vehicle_CurrentLocation_GNSSReceiver and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleCurrentLocationGnssReceiverPublisher : public IVehicleCurrentLocationGnssReceiverPublisher
{
public:
    /**
    * Implementation of IVehicleCurrentLocationGnssReceiverPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleCurrentLocationGnssReceiverSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleCurrentLocationGnssReceiverPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleCurrentLocationGnssReceiverSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleCurrentLocationGnssReceiverPublisher::subscribeToFixTypeChanged
    */
    long subscribeToFixTypeChanged(VehicleCurrentLocationGnssReceiverFixTypePropertyCb callback) override;
    /**
    * Implementation of IVehicleCurrentLocationGnssReceiverPublisher::subscribeToFixTypeChanged
    */
    void unsubscribeFromFixTypeChanged(long handleId) override;

    /**
    * Implementation of IVehicleCurrentLocationGnssReceiverPublisher::publishFixTypeChanged
    */
    void publishFixTypeChanged(Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum FixType) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_CurrentLocation_GNSSReceiver
    std::vector<std::reference_wrapper<IVehicleCurrentLocationGnssReceiverSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the FixType change.
    std::atomic<long> m_fixTypeChangedCallbackNextId {0};
    // Subscribed callbacks for the FixType change.
    std::map<long, VehicleCurrentLocationGnssReceiverFixTypePropertyCb> m_fixTypeCallbacks;
    // Mutex for m_fixTypeCallbacks
    mutable std::shared_timed_mutex m_fixTypeCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
