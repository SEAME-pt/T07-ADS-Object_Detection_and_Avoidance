#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclemotionmanagementbrakeaxlerow1wheelright.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleMotionManagementBrakeAxleRow1WheelRightPublisher.
 * Use this class to store clients of the Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleMotionManagementBrakeAxleRow1WheelRightPublisher : public IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher
{
public:
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleMotionManagementBrakeAxleRow1WheelRightSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleMotionManagementBrakeAxleRow1WheelRightSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::subscribeToTorqueChanged
    */
    long subscribeToTorqueChanged(VehicleMotionManagementBrakeAxleRow1WheelRightTorquePropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::subscribeToTorqueChanged
    */
    void unsubscribeFromTorqueChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::subscribeToTorqueArbitratedChanged
    */
    long subscribeToTorqueArbitratedChanged(VehicleMotionManagementBrakeAxleRow1WheelRightTorqueArbitratedPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::subscribeToTorqueArbitratedChanged
    */
    void unsubscribeFromTorqueArbitratedChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::subscribeToTorqueFrictionMaximumChanged
    */
    long subscribeToTorqueFrictionMaximumChanged(VehicleMotionManagementBrakeAxleRow1WheelRightTorqueFrictionMaximumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::subscribeToTorqueFrictionMaximumChanged
    */
    void unsubscribeFromTorqueFrictionMaximumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::subscribeToTorqueFrictionMinimumChanged
    */
    long subscribeToTorqueFrictionMinimumChanged(VehicleMotionManagementBrakeAxleRow1WheelRightTorqueFrictionMinimumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::subscribeToTorqueFrictionMinimumChanged
    */
    void unsubscribeFromTorqueFrictionMinimumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::subscribeToOmegaUpperChanged
    */
    long subscribeToOmegaUpperChanged(VehicleMotionManagementBrakeAxleRow1WheelRightOmegaUpperPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::subscribeToOmegaUpperChanged
    */
    void unsubscribeFromOmegaUpperChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::subscribeToOmegaLowerChanged
    */
    long subscribeToOmegaLowerChanged(VehicleMotionManagementBrakeAxleRow1WheelRightOmegaLowerPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::subscribeToOmegaLowerChanged
    */
    void unsubscribeFromOmegaLowerChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::publishTorqueChanged
    */
    void publishTorqueChanged(int32_t Torque) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::publishTorqueArbitratedChanged
    */
    void publishTorqueArbitratedChanged(int32_t TorqueArbitrated) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::publishTorqueFrictionMaximumChanged
    */
    void publishTorqueFrictionMaximumChanged(int32_t TorqueFrictionMaximum) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::publishTorqueFrictionMinimumChanged
    */
    void publishTorqueFrictionMinimumChanged(int32_t TorqueFrictionMinimum) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::publishOmegaUpperChanged
    */
    void publishOmegaUpperChanged(int32_t OmegaUpper) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow1WheelRightPublisher::publishOmegaLowerChanged
    */
    void publishOmegaLowerChanged(int32_t OmegaLower) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_MotionManagement_Brake_Axle_Row1_Wheel_Right
    std::vector<std::reference_wrapper<IVehicleMotionManagementBrakeAxleRow1WheelRightSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Torque change.
    std::atomic<long> m_torqueChangedCallbackNextId {0};
    // Subscribed callbacks for the Torque change.
    std::map<long, VehicleMotionManagementBrakeAxleRow1WheelRightTorquePropertyCb> m_torqueCallbacks;
    // Mutex for m_torqueCallbacks
    mutable std::shared_timed_mutex m_torqueCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueArbitrated change.
    std::atomic<long> m_torqueArbitratedChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueArbitrated change.
    std::map<long, VehicleMotionManagementBrakeAxleRow1WheelRightTorqueArbitratedPropertyCb> m_torqueArbitratedCallbacks;
    // Mutex for m_torqueArbitratedCallbacks
    mutable std::shared_timed_mutex m_torqueArbitratedCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueFrictionMaximum change.
    std::atomic<long> m_torqueFrictionMaximumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueFrictionMaximum change.
    std::map<long, VehicleMotionManagementBrakeAxleRow1WheelRightTorqueFrictionMaximumPropertyCb> m_torqueFrictionMaximumCallbacks;
    // Mutex for m_torqueFrictionMaximumCallbacks
    mutable std::shared_timed_mutex m_torqueFrictionMaximumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueFrictionMinimum change.
    std::atomic<long> m_torqueFrictionMinimumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueFrictionMinimum change.
    std::map<long, VehicleMotionManagementBrakeAxleRow1WheelRightTorqueFrictionMinimumPropertyCb> m_torqueFrictionMinimumCallbacks;
    // Mutex for m_torqueFrictionMinimumCallbacks
    mutable std::shared_timed_mutex m_torqueFrictionMinimumCallbacksMutex;
    // Next free unique identifier to subscribe for the OmegaUpper change.
    std::atomic<long> m_omegaUpperChangedCallbackNextId {0};
    // Subscribed callbacks for the OmegaUpper change.
    std::map<long, VehicleMotionManagementBrakeAxleRow1WheelRightOmegaUpperPropertyCb> m_omegaUpperCallbacks;
    // Mutex for m_omegaUpperCallbacks
    mutable std::shared_timed_mutex m_omegaUpperCallbacksMutex;
    // Next free unique identifier to subscribe for the OmegaLower change.
    std::atomic<long> m_omegaLowerChangedCallbackNextId {0};
    // Subscribed callbacks for the OmegaLower change.
    std::map<long, VehicleMotionManagementBrakeAxleRow1WheelRightOmegaLowerPropertyCb> m_omegaLowerCallbacks;
    // Mutex for m_omegaLowerCallbacks
    mutable std::shared_timed_mutex m_omegaLowerCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
