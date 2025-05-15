#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclemotionmanagementsteeringsteeringwheel.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleMotionManagementSteeringSteeringWheelPublisher.
 * Use this class to store clients of the Vehicle_MotionManagement_Steering_SteeringWheel and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleMotionManagementSteeringSteeringWheelPublisher : public IVehicleMotionManagementSteeringSteeringWheelPublisher
{
public:
    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleMotionManagementSteeringSteeringWheelSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleMotionManagementSteeringSteeringWheelSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToAngleChanged
    */
    long subscribeToAngleChanged(VehicleMotionManagementSteeringSteeringWheelAnglePropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToAngleChanged
    */
    void unsubscribeFromAngleChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToAngleTargetChanged
    */
    long subscribeToAngleTargetChanged(VehicleMotionManagementSteeringSteeringWheelAngleTargetPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToAngleTargetChanged
    */
    void unsubscribeFromAngleTargetChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToAngleTargetModeChanged
    */
    long subscribeToAngleTargetModeChanged(VehicleMotionManagementSteeringSteeringWheelAngleTargetModePropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToAngleTargetModeChanged
    */
    void unsubscribeFromAngleTargetModeChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToTorqueOffsetTargetChanged
    */
    long subscribeToTorqueOffsetTargetChanged(VehicleMotionManagementSteeringSteeringWheelTorqueOffsetTargetPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToTorqueOffsetTargetChanged
    */
    void unsubscribeFromTorqueOffsetTargetChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToTorqueOffsetTargetModeChanged
    */
    long subscribeToTorqueOffsetTargetModeChanged(VehicleMotionManagementSteeringSteeringWheelTorqueOffsetTargetModePropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToTorqueOffsetTargetModeChanged
    */
    void unsubscribeFromTorqueOffsetTargetModeChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToTorqueTargetChanged
    */
    long subscribeToTorqueTargetChanged(VehicleMotionManagementSteeringSteeringWheelTorqueTargetPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToTorqueTargetChanged
    */
    void unsubscribeFromTorqueTargetChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToTorqueTargetModeChanged
    */
    long subscribeToTorqueTargetModeChanged(VehicleMotionManagementSteeringSteeringWheelTorqueTargetModePropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToTorqueTargetModeChanged
    */
    void unsubscribeFromTorqueTargetModeChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToTorqueChanged
    */
    long subscribeToTorqueChanged(VehicleMotionManagementSteeringSteeringWheelTorquePropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToTorqueChanged
    */
    void unsubscribeFromTorqueChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::publishAngleChanged
    */
    void publishAngleChanged(int32_t Angle) const override;
    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::publishAngleTargetChanged
    */
    void publishAngleTargetChanged(int32_t AngleTarget) const override;
    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::publishAngleTargetModeChanged
    */
    void publishAngleTargetModeChanged(int32_t AngleTargetMode) const override;
    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::publishTorqueOffsetTargetChanged
    */
    void publishTorqueOffsetTargetChanged(int32_t TorqueOffsetTarget) const override;
    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::publishTorqueOffsetTargetModeChanged
    */
    void publishTorqueOffsetTargetModeChanged(int32_t TorqueOffsetTargetMode) const override;
    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::publishTorqueTargetChanged
    */
    void publishTorqueTargetChanged(int32_t TorqueTarget) const override;
    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::publishTorqueTargetModeChanged
    */
    void publishTorqueTargetModeChanged(int32_t TorqueTargetMode) const override;
    /**
    * Implementation of IVehicleMotionManagementSteeringSteeringWheelPublisher::publishTorqueChanged
    */
    void publishTorqueChanged(int32_t Torque) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_MotionManagement_Steering_SteeringWheel
    std::vector<std::reference_wrapper<IVehicleMotionManagementSteeringSteeringWheelSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Angle change.
    std::atomic<long> m_angleChangedCallbackNextId {0};
    // Subscribed callbacks for the Angle change.
    std::map<long, VehicleMotionManagementSteeringSteeringWheelAnglePropertyCb> m_angleCallbacks;
    // Mutex for m_angleCallbacks
    mutable std::shared_timed_mutex m_angleCallbacksMutex;
    // Next free unique identifier to subscribe for the AngleTarget change.
    std::atomic<long> m_angleTargetChangedCallbackNextId {0};
    // Subscribed callbacks for the AngleTarget change.
    std::map<long, VehicleMotionManagementSteeringSteeringWheelAngleTargetPropertyCb> m_angleTargetCallbacks;
    // Mutex for m_angleTargetCallbacks
    mutable std::shared_timed_mutex m_angleTargetCallbacksMutex;
    // Next free unique identifier to subscribe for the AngleTargetMode change.
    std::atomic<long> m_angleTargetModeChangedCallbackNextId {0};
    // Subscribed callbacks for the AngleTargetMode change.
    std::map<long, VehicleMotionManagementSteeringSteeringWheelAngleTargetModePropertyCb> m_angleTargetModeCallbacks;
    // Mutex for m_angleTargetModeCallbacks
    mutable std::shared_timed_mutex m_angleTargetModeCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueOffsetTarget change.
    std::atomic<long> m_torqueOffsetTargetChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueOffsetTarget change.
    std::map<long, VehicleMotionManagementSteeringSteeringWheelTorqueOffsetTargetPropertyCb> m_torqueOffsetTargetCallbacks;
    // Mutex for m_torqueOffsetTargetCallbacks
    mutable std::shared_timed_mutex m_torqueOffsetTargetCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueOffsetTargetMode change.
    std::atomic<long> m_torqueOffsetTargetModeChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueOffsetTargetMode change.
    std::map<long, VehicleMotionManagementSteeringSteeringWheelTorqueOffsetTargetModePropertyCb> m_torqueOffsetTargetModeCallbacks;
    // Mutex for m_torqueOffsetTargetModeCallbacks
    mutable std::shared_timed_mutex m_torqueOffsetTargetModeCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueTarget change.
    std::atomic<long> m_torqueTargetChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueTarget change.
    std::map<long, VehicleMotionManagementSteeringSteeringWheelTorqueTargetPropertyCb> m_torqueTargetCallbacks;
    // Mutex for m_torqueTargetCallbacks
    mutable std::shared_timed_mutex m_torqueTargetCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueTargetMode change.
    std::atomic<long> m_torqueTargetModeChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueTargetMode change.
    std::map<long, VehicleMotionManagementSteeringSteeringWheelTorqueTargetModePropertyCb> m_torqueTargetModeCallbacks;
    // Mutex for m_torqueTargetModeCallbacks
    mutable std::shared_timed_mutex m_torqueTargetModeCallbacksMutex;
    // Next free unique identifier to subscribe for the Torque change.
    std::atomic<long> m_torqueChangedCallbackNextId {0};
    // Subscribed callbacks for the Torque change.
    std::map<long, VehicleMotionManagementSteeringSteeringWheelTorquePropertyCb> m_torqueCallbacks;
    // Mutex for m_torqueCallbacks
    mutable std::shared_timed_mutex m_torqueCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
