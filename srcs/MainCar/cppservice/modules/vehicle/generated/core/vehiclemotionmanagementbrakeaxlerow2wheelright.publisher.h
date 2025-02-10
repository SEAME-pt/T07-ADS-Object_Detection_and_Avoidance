#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclemotionmanagementbrakeaxlerow2wheelright.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleMotionManagementBrakeAxleRow2WheelRightPublisher.
 * Use this class to store clients of the Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleMotionManagementBrakeAxleRow2WheelRightPublisher : public IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher
{
public:
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleMotionManagementBrakeAxleRow2WheelRightSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleMotionManagementBrakeAxleRow2WheelRightSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::subscribeToTorqueChanged
    */
    long subscribeToTorqueChanged(VehicleMotionManagementBrakeAxleRow2WheelRightTorquePropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::subscribeToTorqueChanged
    */
    void unsubscribeFromTorqueChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::subscribeToTorqueArbitratedChanged
    */
    long subscribeToTorqueArbitratedChanged(VehicleMotionManagementBrakeAxleRow2WheelRightTorqueArbitratedPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::subscribeToTorqueArbitratedChanged
    */
    void unsubscribeFromTorqueArbitratedChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::subscribeToTorqueFrictionMaximumChanged
    */
    long subscribeToTorqueFrictionMaximumChanged(VehicleMotionManagementBrakeAxleRow2WheelRightTorqueFrictionMaximumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::subscribeToTorqueFrictionMaximumChanged
    */
    void unsubscribeFromTorqueFrictionMaximumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::subscribeToTorqueFrictionMinimumChanged
    */
    long subscribeToTorqueFrictionMinimumChanged(VehicleMotionManagementBrakeAxleRow2WheelRightTorqueFrictionMinimumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::subscribeToTorqueFrictionMinimumChanged
    */
    void unsubscribeFromTorqueFrictionMinimumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::subscribeToOmegaUpperChanged
    */
    long subscribeToOmegaUpperChanged(VehicleMotionManagementBrakeAxleRow2WheelRightOmegaUpperPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::subscribeToOmegaUpperChanged
    */
    void unsubscribeFromOmegaUpperChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::subscribeToOmegaLowerChanged
    */
    long subscribeToOmegaLowerChanged(VehicleMotionManagementBrakeAxleRow2WheelRightOmegaLowerPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::subscribeToOmegaLowerChanged
    */
    void unsubscribeFromOmegaLowerChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::publishTorqueChanged
    */
    void publishTorqueChanged(int32_t Torque) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::publishTorqueArbitratedChanged
    */
    void publishTorqueArbitratedChanged(int32_t TorqueArbitrated) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::publishTorqueFrictionMaximumChanged
    */
    void publishTorqueFrictionMaximumChanged(int32_t TorqueFrictionMaximum) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::publishTorqueFrictionMinimumChanged
    */
    void publishTorqueFrictionMinimumChanged(int32_t TorqueFrictionMinimum) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::publishOmegaUpperChanged
    */
    void publishOmegaUpperChanged(int32_t OmegaUpper) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelRightPublisher::publishOmegaLowerChanged
    */
    void publishOmegaLowerChanged(int32_t OmegaLower) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Right
    std::vector<std::reference_wrapper<IVehicleMotionManagementBrakeAxleRow2WheelRightSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Torque change.
    std::atomic<long> m_torqueChangedCallbackNextId {0};
    // Subscribed callbacks for the Torque change.
    std::map<long, VehicleMotionManagementBrakeAxleRow2WheelRightTorquePropertyCb> m_torqueCallbacks;
    // Mutex for m_torqueCallbacks
    mutable std::shared_timed_mutex m_torqueCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueArbitrated change.
    std::atomic<long> m_torqueArbitratedChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueArbitrated change.
    std::map<long, VehicleMotionManagementBrakeAxleRow2WheelRightTorqueArbitratedPropertyCb> m_torqueArbitratedCallbacks;
    // Mutex for m_torqueArbitratedCallbacks
    mutable std::shared_timed_mutex m_torqueArbitratedCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueFrictionMaximum change.
    std::atomic<long> m_torqueFrictionMaximumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueFrictionMaximum change.
    std::map<long, VehicleMotionManagementBrakeAxleRow2WheelRightTorqueFrictionMaximumPropertyCb> m_torqueFrictionMaximumCallbacks;
    // Mutex for m_torqueFrictionMaximumCallbacks
    mutable std::shared_timed_mutex m_torqueFrictionMaximumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueFrictionMinimum change.
    std::atomic<long> m_torqueFrictionMinimumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueFrictionMinimum change.
    std::map<long, VehicleMotionManagementBrakeAxleRow2WheelRightTorqueFrictionMinimumPropertyCb> m_torqueFrictionMinimumCallbacks;
    // Mutex for m_torqueFrictionMinimumCallbacks
    mutable std::shared_timed_mutex m_torqueFrictionMinimumCallbacksMutex;
    // Next free unique identifier to subscribe for the OmegaUpper change.
    std::atomic<long> m_omegaUpperChangedCallbackNextId {0};
    // Subscribed callbacks for the OmegaUpper change.
    std::map<long, VehicleMotionManagementBrakeAxleRow2WheelRightOmegaUpperPropertyCb> m_omegaUpperCallbacks;
    // Mutex for m_omegaUpperCallbacks
    mutable std::shared_timed_mutex m_omegaUpperCallbacksMutex;
    // Next free unique identifier to subscribe for the OmegaLower change.
    std::atomic<long> m_omegaLowerChangedCallbackNextId {0};
    // Subscribed callbacks for the OmegaLower change.
    std::map<long, VehicleMotionManagementBrakeAxleRow2WheelRightOmegaLowerPropertyCb> m_omegaLowerCallbacks;
    // Mutex for m_omegaLowerCallbacks
    mutable std::shared_timed_mutex m_omegaLowerCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
