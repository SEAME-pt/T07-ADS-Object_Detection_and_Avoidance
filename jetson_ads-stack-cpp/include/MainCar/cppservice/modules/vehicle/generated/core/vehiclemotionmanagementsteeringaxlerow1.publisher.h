#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclemotionmanagementsteeringaxlerow1.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleMotionManagementSteeringAxleRow1Publisher.
 * Use this class to store clients of the Vehicle_MotionManagement_Steering_Axle_Row1 and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleMotionManagementSteeringAxleRow1Publisher : public IVehicleMotionManagementSteeringAxleRow1Publisher
{
public:
    /**
    * Implementation of IVehicleMotionManagementSteeringAxleRow1Publisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleMotionManagementSteeringAxleRow1Subscriber& subscriber) override;
    /**
    * Implementation of IVehicleMotionManagementSteeringAxleRow1Publisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleMotionManagementSteeringAxleRow1Subscriber& subscriber) override;

    /**
    * Implementation of IVehicleMotionManagementSteeringAxleRow1Publisher::subscribeToRackPositionOffsetTargetChanged
    */
    long subscribeToRackPositionOffsetTargetChanged(VehicleMotionManagementSteeringAxleRow1RackPositionOffsetTargetPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementSteeringAxleRow1Publisher::subscribeToRackPositionOffsetTargetChanged
    */
    void unsubscribeFromRackPositionOffsetTargetChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementSteeringAxleRow1Publisher::subscribeToRackPositionOffsetTargetModeChanged
    */
    long subscribeToRackPositionOffsetTargetModeChanged(VehicleMotionManagementSteeringAxleRow1RackPositionOffsetTargetModePropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementSteeringAxleRow1Publisher::subscribeToRackPositionOffsetTargetModeChanged
    */
    void unsubscribeFromRackPositionOffsetTargetModeChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementSteeringAxleRow1Publisher::subscribeToRackPositionChanged
    */
    long subscribeToRackPositionChanged(VehicleMotionManagementSteeringAxleRow1RackPositionPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementSteeringAxleRow1Publisher::subscribeToRackPositionChanged
    */
    void unsubscribeFromRackPositionChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementSteeringAxleRow1Publisher::subscribeToRackPositionTargetChanged
    */
    long subscribeToRackPositionTargetChanged(VehicleMotionManagementSteeringAxleRow1RackPositionTargetPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementSteeringAxleRow1Publisher::subscribeToRackPositionTargetChanged
    */
    void unsubscribeFromRackPositionTargetChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementSteeringAxleRow1Publisher::subscribeToRackPositionTargetModeChanged
    */
    long subscribeToRackPositionTargetModeChanged(VehicleMotionManagementSteeringAxleRow1RackPositionTargetModePropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementSteeringAxleRow1Publisher::subscribeToRackPositionTargetModeChanged
    */
    void unsubscribeFromRackPositionTargetModeChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementSteeringAxleRow1Publisher::publishRackPositionOffsetTargetChanged
    */
    void publishRackPositionOffsetTargetChanged(int32_t RackPositionOffsetTarget) const override;
    /**
    * Implementation of IVehicleMotionManagementSteeringAxleRow1Publisher::publishRackPositionOffsetTargetModeChanged
    */
    void publishRackPositionOffsetTargetModeChanged(int32_t RackPositionOffsetTargetMode) const override;
    /**
    * Implementation of IVehicleMotionManagementSteeringAxleRow1Publisher::publishRackPositionChanged
    */
    void publishRackPositionChanged(int32_t RackPosition) const override;
    /**
    * Implementation of IVehicleMotionManagementSteeringAxleRow1Publisher::publishRackPositionTargetChanged
    */
    void publishRackPositionTargetChanged(int32_t RackPositionTarget) const override;
    /**
    * Implementation of IVehicleMotionManagementSteeringAxleRow1Publisher::publishRackPositionTargetModeChanged
    */
    void publishRackPositionTargetModeChanged(int32_t RackPositionTargetMode) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_MotionManagement_Steering_Axle_Row1
    std::vector<std::reference_wrapper<IVehicleMotionManagementSteeringAxleRow1Subscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the RackPositionOffsetTarget change.
    std::atomic<long> m_rackPositionOffsetTargetChangedCallbackNextId {0};
    // Subscribed callbacks for the RackPositionOffsetTarget change.
    std::map<long, VehicleMotionManagementSteeringAxleRow1RackPositionOffsetTargetPropertyCb> m_rackPositionOffsetTargetCallbacks;
    // Mutex for m_rackPositionOffsetTargetCallbacks
    mutable std::shared_timed_mutex m_rackPositionOffsetTargetCallbacksMutex;
    // Next free unique identifier to subscribe for the RackPositionOffsetTargetMode change.
    std::atomic<long> m_rackPositionOffsetTargetModeChangedCallbackNextId {0};
    // Subscribed callbacks for the RackPositionOffsetTargetMode change.
    std::map<long, VehicleMotionManagementSteeringAxleRow1RackPositionOffsetTargetModePropertyCb> m_rackPositionOffsetTargetModeCallbacks;
    // Mutex for m_rackPositionOffsetTargetModeCallbacks
    mutable std::shared_timed_mutex m_rackPositionOffsetTargetModeCallbacksMutex;
    // Next free unique identifier to subscribe for the RackPosition change.
    std::atomic<long> m_rackPositionChangedCallbackNextId {0};
    // Subscribed callbacks for the RackPosition change.
    std::map<long, VehicleMotionManagementSteeringAxleRow1RackPositionPropertyCb> m_rackPositionCallbacks;
    // Mutex for m_rackPositionCallbacks
    mutable std::shared_timed_mutex m_rackPositionCallbacksMutex;
    // Next free unique identifier to subscribe for the RackPositionTarget change.
    std::atomic<long> m_rackPositionTargetChangedCallbackNextId {0};
    // Subscribed callbacks for the RackPositionTarget change.
    std::map<long, VehicleMotionManagementSteeringAxleRow1RackPositionTargetPropertyCb> m_rackPositionTargetCallbacks;
    // Mutex for m_rackPositionTargetCallbacks
    mutable std::shared_timed_mutex m_rackPositionTargetCallbacksMutex;
    // Next free unique identifier to subscribe for the RackPositionTargetMode change.
    std::atomic<long> m_rackPositionTargetModeChangedCallbackNextId {0};
    // Subscribed callbacks for the RackPositionTargetMode change.
    std::map<long, VehicleMotionManagementSteeringAxleRow1RackPositionTargetModePropertyCb> m_rackPositionTargetModeCallbacks;
    // Mutex for m_rackPositionTargetModeCallbacks
    mutable std::shared_timed_mutex m_rackPositionTargetModeCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
