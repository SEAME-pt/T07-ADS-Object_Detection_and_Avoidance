#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclecurrentlocationgnssreceivermountingposition.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleCurrentLocationGnssReceiverMountingPositionPublisher.
 * Use this class to store clients of the Vehicle_CurrentLocation_GNSSReceiver_MountingPosition and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleCurrentLocationGnssReceiverMountingPositionPublisher : public IVehicleCurrentLocationGnssReceiverMountingPositionPublisher
{
public:
    /**
    * Implementation of IVehicleCurrentLocationGnssReceiverMountingPositionPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleCurrentLocationGnssReceiverMountingPositionPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleCurrentLocationGnssReceiverMountingPositionPublisher::subscribeToXChanged
    */
    long subscribeToXChanged(VehicleCurrentLocationGnssReceiverMountingPositionXPropertyCb callback) override;
    /**
    * Implementation of IVehicleCurrentLocationGnssReceiverMountingPositionPublisher::subscribeToXChanged
    */
    void unsubscribeFromXChanged(long handleId) override;

    /**
    * Implementation of IVehicleCurrentLocationGnssReceiverMountingPositionPublisher::subscribeToYChanged
    */
    long subscribeToYChanged(VehicleCurrentLocationGnssReceiverMountingPositionYPropertyCb callback) override;
    /**
    * Implementation of IVehicleCurrentLocationGnssReceiverMountingPositionPublisher::subscribeToYChanged
    */
    void unsubscribeFromYChanged(long handleId) override;

    /**
    * Implementation of IVehicleCurrentLocationGnssReceiverMountingPositionPublisher::subscribeToZChanged
    */
    long subscribeToZChanged(VehicleCurrentLocationGnssReceiverMountingPositionZPropertyCb callback) override;
    /**
    * Implementation of IVehicleCurrentLocationGnssReceiverMountingPositionPublisher::subscribeToZChanged
    */
    void unsubscribeFromZChanged(long handleId) override;

    /**
    * Implementation of IVehicleCurrentLocationGnssReceiverMountingPositionPublisher::publishXChanged
    */
    void publishXChanged(int32_t X) const override;
    /**
    * Implementation of IVehicleCurrentLocationGnssReceiverMountingPositionPublisher::publishYChanged
    */
    void publishYChanged(int32_t Y) const override;
    /**
    * Implementation of IVehicleCurrentLocationGnssReceiverMountingPositionPublisher::publishZChanged
    */
    void publishZChanged(int32_t Z) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_CurrentLocation_GNSSReceiver_MountingPosition
    std::vector<std::reference_wrapper<IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the X change.
    std::atomic<long> m_xChangedCallbackNextId {0};
    // Subscribed callbacks for the X change.
    std::map<long, VehicleCurrentLocationGnssReceiverMountingPositionXPropertyCb> m_xCallbacks;
    // Mutex for m_xCallbacks
    mutable std::shared_timed_mutex m_xCallbacksMutex;
    // Next free unique identifier to subscribe for the Y change.
    std::atomic<long> m_yChangedCallbackNextId {0};
    // Subscribed callbacks for the Y change.
    std::map<long, VehicleCurrentLocationGnssReceiverMountingPositionYPropertyCb> m_yCallbacks;
    // Mutex for m_yCallbacks
    mutable std::shared_timed_mutex m_yCallbacksMutex;
    // Next free unique identifier to subscribe for the Z change.
    std::atomic<long> m_zChangedCallbackNextId {0};
    // Subscribed callbacks for the Z change.
    std::map<long, VehicleCurrentLocationGnssReceiverMountingPositionZPropertyCb> m_zCallbacks;
    // Mutex for m_zCallbacks
    mutable std::shared_timed_mutex m_zCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
