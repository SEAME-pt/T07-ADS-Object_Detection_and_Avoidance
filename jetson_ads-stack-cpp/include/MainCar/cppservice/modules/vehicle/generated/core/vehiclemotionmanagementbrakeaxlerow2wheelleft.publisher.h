#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclemotionmanagementbrakeaxlerow2wheelleft.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleMotionManagementBrakeAxleRow2WheelLeftPublisher.
 * Use this class to store clients of the Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Left and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleMotionManagementBrakeAxleRow2WheelLeftPublisher : public IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher
{
public:
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleMotionManagementBrakeAxleRow2WheelLeftSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleMotionManagementBrakeAxleRow2WheelLeftSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::subscribeToTorqueChanged
    */
    long subscribeToTorqueChanged(VehicleMotionManagementBrakeAxleRow2WheelLeftTorquePropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::subscribeToTorqueChanged
    */
    void unsubscribeFromTorqueChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::subscribeToTorqueArbitratedChanged
    */
    long subscribeToTorqueArbitratedChanged(VehicleMotionManagementBrakeAxleRow2WheelLeftTorqueArbitratedPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::subscribeToTorqueArbitratedChanged
    */
    void unsubscribeFromTorqueArbitratedChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::subscribeToTorqueFrictionMaximumChanged
    */
    long subscribeToTorqueFrictionMaximumChanged(VehicleMotionManagementBrakeAxleRow2WheelLeftTorqueFrictionMaximumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::subscribeToTorqueFrictionMaximumChanged
    */
    void unsubscribeFromTorqueFrictionMaximumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::subscribeToTorqueFrictionMinimumChanged
    */
    long subscribeToTorqueFrictionMinimumChanged(VehicleMotionManagementBrakeAxleRow2WheelLeftTorqueFrictionMinimumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::subscribeToTorqueFrictionMinimumChanged
    */
    void unsubscribeFromTorqueFrictionMinimumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::subscribeToOmegaUpperChanged
    */
    long subscribeToOmegaUpperChanged(VehicleMotionManagementBrakeAxleRow2WheelLeftOmegaUpperPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::subscribeToOmegaUpperChanged
    */
    void unsubscribeFromOmegaUpperChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::subscribeToOmegaLowerChanged
    */
    long subscribeToOmegaLowerChanged(VehicleMotionManagementBrakeAxleRow2WheelLeftOmegaLowerPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::subscribeToOmegaLowerChanged
    */
    void unsubscribeFromOmegaLowerChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::publishTorqueChanged
    */
    void publishTorqueChanged(int32_t Torque) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::publishTorqueArbitratedChanged
    */
    void publishTorqueArbitratedChanged(int32_t TorqueArbitrated) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::publishTorqueFrictionMaximumChanged
    */
    void publishTorqueFrictionMaximumChanged(int32_t TorqueFrictionMaximum) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::publishTorqueFrictionMinimumChanged
    */
    void publishTorqueFrictionMinimumChanged(int32_t TorqueFrictionMinimum) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::publishOmegaUpperChanged
    */
    void publishOmegaUpperChanged(int32_t OmegaUpper) const override;
    /**
    * Implementation of IVehicleMotionManagementBrakeAxleRow2WheelLeftPublisher::publishOmegaLowerChanged
    */
    void publishOmegaLowerChanged(int32_t OmegaLower) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_MotionManagement_Brake_Axle_Row2_Wheel_Left
    std::vector<std::reference_wrapper<IVehicleMotionManagementBrakeAxleRow2WheelLeftSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Torque change.
    std::atomic<long> m_torqueChangedCallbackNextId {0};
    // Subscribed callbacks for the Torque change.
    std::map<long, VehicleMotionManagementBrakeAxleRow2WheelLeftTorquePropertyCb> m_torqueCallbacks;
    // Mutex for m_torqueCallbacks
    mutable std::shared_timed_mutex m_torqueCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueArbitrated change.
    std::atomic<long> m_torqueArbitratedChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueArbitrated change.
    std::map<long, VehicleMotionManagementBrakeAxleRow2WheelLeftTorqueArbitratedPropertyCb> m_torqueArbitratedCallbacks;
    // Mutex for m_torqueArbitratedCallbacks
    mutable std::shared_timed_mutex m_torqueArbitratedCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueFrictionMaximum change.
    std::atomic<long> m_torqueFrictionMaximumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueFrictionMaximum change.
    std::map<long, VehicleMotionManagementBrakeAxleRow2WheelLeftTorqueFrictionMaximumPropertyCb> m_torqueFrictionMaximumCallbacks;
    // Mutex for m_torqueFrictionMaximumCallbacks
    mutable std::shared_timed_mutex m_torqueFrictionMaximumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueFrictionMinimum change.
    std::atomic<long> m_torqueFrictionMinimumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueFrictionMinimum change.
    std::map<long, VehicleMotionManagementBrakeAxleRow2WheelLeftTorqueFrictionMinimumPropertyCb> m_torqueFrictionMinimumCallbacks;
    // Mutex for m_torqueFrictionMinimumCallbacks
    mutable std::shared_timed_mutex m_torqueFrictionMinimumCallbacksMutex;
    // Next free unique identifier to subscribe for the OmegaUpper change.
    std::atomic<long> m_omegaUpperChangedCallbackNextId {0};
    // Subscribed callbacks for the OmegaUpper change.
    std::map<long, VehicleMotionManagementBrakeAxleRow2WheelLeftOmegaUpperPropertyCb> m_omegaUpperCallbacks;
    // Mutex for m_omegaUpperCallbacks
    mutable std::shared_timed_mutex m_omegaUpperCallbacksMutex;
    // Next free unique identifier to subscribe for the OmegaLower change.
    std::atomic<long> m_omegaLowerChangedCallbackNextId {0};
    // Subscribed callbacks for the OmegaLower change.
    std::map<long, VehicleMotionManagementBrakeAxleRow2WheelLeftOmegaLowerPropertyCb> m_omegaLowerCallbacks;
    // Mutex for m_omegaLowerCallbacks
    mutable std::shared_timed_mutex m_omegaLowerCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
