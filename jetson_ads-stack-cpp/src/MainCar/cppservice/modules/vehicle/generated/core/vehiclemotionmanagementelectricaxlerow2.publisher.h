#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclemotionmanagementelectricaxlerow2.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleMotionManagementElectricAxleRow2Publisher.
 * Use this class to store clients of the Vehicle_MotionManagement_ElectricAxle_Row2 and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleMotionManagementElectricAxleRow2Publisher : public IVehicleMotionManagementElectricAxleRow2Publisher
{
public:
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleMotionManagementElectricAxleRow2Subscriber& subscriber) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleMotionManagementElectricAxleRow2Subscriber& subscriber) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueChanged
    */
    long subscribeToTorqueChanged(VehicleMotionManagementElectricAxleRow2TorquePropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueChanged
    */
    void unsubscribeFromTorqueChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueTargetChanged
    */
    long subscribeToTorqueTargetChanged(VehicleMotionManagementElectricAxleRow2TorqueTargetPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueTargetChanged
    */
    void unsubscribeFromTorqueTargetChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueMaximumChanged
    */
    long subscribeToTorqueMaximumChanged(VehicleMotionManagementElectricAxleRow2TorqueMaximumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueMaximumChanged
    */
    void unsubscribeFromTorqueMaximumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueMinimumChanged
    */
    long subscribeToTorqueMinimumChanged(VehicleMotionManagementElectricAxleRow2TorqueMinimumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueMinimumChanged
    */
    void unsubscribeFromTorqueMinimumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueLongTermMaximumChanged
    */
    long subscribeToTorqueLongTermMaximumChanged(VehicleMotionManagementElectricAxleRow2TorqueLongTermMaximumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueLongTermMaximumChanged
    */
    void unsubscribeFromTorqueLongTermMaximumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueLongTermMinimumChanged
    */
    long subscribeToTorqueLongTermMinimumChanged(VehicleMotionManagementElectricAxleRow2TorqueLongTermMinimumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueLongTermMinimumChanged
    */
    void unsubscribeFromTorqueLongTermMinimumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueShortTermMaximumChanged
    */
    long subscribeToTorqueShortTermMaximumChanged(VehicleMotionManagementElectricAxleRow2TorqueShortTermMaximumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueShortTermMaximumChanged
    */
    void unsubscribeFromTorqueShortTermMaximumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueShortTermMinimumChanged
    */
    long subscribeToTorqueShortTermMinimumChanged(VehicleMotionManagementElectricAxleRow2TorqueShortTermMinimumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueShortTermMinimumChanged
    */
    void unsubscribeFromTorqueShortTermMinimumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueMaximumLimitChanged
    */
    long subscribeToTorqueMaximumLimitChanged(VehicleMotionManagementElectricAxleRow2TorqueMaximumLimitPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueMaximumLimitChanged
    */
    void unsubscribeFromTorqueMaximumLimitChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueMinimumLimitChanged
    */
    long subscribeToTorqueMinimumLimitChanged(VehicleMotionManagementElectricAxleRow2TorqueMinimumLimitPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueMinimumLimitChanged
    */
    void unsubscribeFromTorqueMinimumLimitChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToRotationalSpeedChanged
    */
    long subscribeToRotationalSpeedChanged(VehicleMotionManagementElectricAxleRow2RotationalSpeedPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToRotationalSpeedChanged
    */
    void unsubscribeFromRotationalSpeedChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToRotationalSpeedTargetChanged
    */
    long subscribeToRotationalSpeedTargetChanged(VehicleMotionManagementElectricAxleRow2RotationalSpeedTargetPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToRotationalSpeedTargetChanged
    */
    void unsubscribeFromRotationalSpeedTargetChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToRotationalSpeedMaximumLimitChanged
    */
    long subscribeToRotationalSpeedMaximumLimitChanged(VehicleMotionManagementElectricAxleRow2RotationalSpeedMaximumLimitPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToRotationalSpeedMaximumLimitChanged
    */
    void unsubscribeFromRotationalSpeedMaximumLimitChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToRotationalSpeedMinimumLimitChanged
    */
    long subscribeToRotationalSpeedMinimumLimitChanged(VehicleMotionManagementElectricAxleRow2RotationalSpeedMinimumLimitPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::subscribeToRotationalSpeedMinimumLimitChanged
    */
    void unsubscribeFromRotationalSpeedMinimumLimitChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::publishTorqueChanged
    */
    void publishTorqueChanged(int32_t Torque) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::publishTorqueTargetChanged
    */
    void publishTorqueTargetChanged(int32_t TorqueTarget) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::publishTorqueMaximumChanged
    */
    void publishTorqueMaximumChanged(int32_t TorqueMaximum) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::publishTorqueMinimumChanged
    */
    void publishTorqueMinimumChanged(int32_t TorqueMinimum) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::publishTorqueLongTermMaximumChanged
    */
    void publishTorqueLongTermMaximumChanged(int32_t TorqueLongTermMaximum) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::publishTorqueLongTermMinimumChanged
    */
    void publishTorqueLongTermMinimumChanged(int32_t TorqueLongTermMinimum) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::publishTorqueShortTermMaximumChanged
    */
    void publishTorqueShortTermMaximumChanged(int32_t TorqueShortTermMaximum) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::publishTorqueShortTermMinimumChanged
    */
    void publishTorqueShortTermMinimumChanged(int32_t TorqueShortTermMinimum) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::publishTorqueMaximumLimitChanged
    */
    void publishTorqueMaximumLimitChanged(int32_t TorqueMaximumLimit) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::publishTorqueMinimumLimitChanged
    */
    void publishTorqueMinimumLimitChanged(int32_t TorqueMinimumLimit) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::publishRotationalSpeedChanged
    */
    void publishRotationalSpeedChanged(int32_t RotationalSpeed) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::publishRotationalSpeedTargetChanged
    */
    void publishRotationalSpeedTargetChanged(int32_t RotationalSpeedTarget) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::publishRotationalSpeedMaximumLimitChanged
    */
    void publishRotationalSpeedMaximumLimitChanged(int32_t RotationalSpeedMaximumLimit) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow2Publisher::publishRotationalSpeedMinimumLimitChanged
    */
    void publishRotationalSpeedMinimumLimitChanged(int32_t RotationalSpeedMinimumLimit) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_MotionManagement_ElectricAxle_Row2
    std::vector<std::reference_wrapper<IVehicleMotionManagementElectricAxleRow2Subscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Torque change.
    std::atomic<long> m_torqueChangedCallbackNextId {0};
    // Subscribed callbacks for the Torque change.
    std::map<long, VehicleMotionManagementElectricAxleRow2TorquePropertyCb> m_torqueCallbacks;
    // Mutex for m_torqueCallbacks
    mutable std::shared_timed_mutex m_torqueCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueTarget change.
    std::atomic<long> m_torqueTargetChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueTarget change.
    std::map<long, VehicleMotionManagementElectricAxleRow2TorqueTargetPropertyCb> m_torqueTargetCallbacks;
    // Mutex for m_torqueTargetCallbacks
    mutable std::shared_timed_mutex m_torqueTargetCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueMaximum change.
    std::atomic<long> m_torqueMaximumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueMaximum change.
    std::map<long, VehicleMotionManagementElectricAxleRow2TorqueMaximumPropertyCb> m_torqueMaximumCallbacks;
    // Mutex for m_torqueMaximumCallbacks
    mutable std::shared_timed_mutex m_torqueMaximumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueMinimum change.
    std::atomic<long> m_torqueMinimumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueMinimum change.
    std::map<long, VehicleMotionManagementElectricAxleRow2TorqueMinimumPropertyCb> m_torqueMinimumCallbacks;
    // Mutex for m_torqueMinimumCallbacks
    mutable std::shared_timed_mutex m_torqueMinimumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueLongTermMaximum change.
    std::atomic<long> m_torqueLongTermMaximumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueLongTermMaximum change.
    std::map<long, VehicleMotionManagementElectricAxleRow2TorqueLongTermMaximumPropertyCb> m_torqueLongTermMaximumCallbacks;
    // Mutex for m_torqueLongTermMaximumCallbacks
    mutable std::shared_timed_mutex m_torqueLongTermMaximumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueLongTermMinimum change.
    std::atomic<long> m_torqueLongTermMinimumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueLongTermMinimum change.
    std::map<long, VehicleMotionManagementElectricAxleRow2TorqueLongTermMinimumPropertyCb> m_torqueLongTermMinimumCallbacks;
    // Mutex for m_torqueLongTermMinimumCallbacks
    mutable std::shared_timed_mutex m_torqueLongTermMinimumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueShortTermMaximum change.
    std::atomic<long> m_torqueShortTermMaximumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueShortTermMaximum change.
    std::map<long, VehicleMotionManagementElectricAxleRow2TorqueShortTermMaximumPropertyCb> m_torqueShortTermMaximumCallbacks;
    // Mutex for m_torqueShortTermMaximumCallbacks
    mutable std::shared_timed_mutex m_torqueShortTermMaximumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueShortTermMinimum change.
    std::atomic<long> m_torqueShortTermMinimumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueShortTermMinimum change.
    std::map<long, VehicleMotionManagementElectricAxleRow2TorqueShortTermMinimumPropertyCb> m_torqueShortTermMinimumCallbacks;
    // Mutex for m_torqueShortTermMinimumCallbacks
    mutable std::shared_timed_mutex m_torqueShortTermMinimumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueMaximumLimit change.
    std::atomic<long> m_torqueMaximumLimitChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueMaximumLimit change.
    std::map<long, VehicleMotionManagementElectricAxleRow2TorqueMaximumLimitPropertyCb> m_torqueMaximumLimitCallbacks;
    // Mutex for m_torqueMaximumLimitCallbacks
    mutable std::shared_timed_mutex m_torqueMaximumLimitCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueMinimumLimit change.
    std::atomic<long> m_torqueMinimumLimitChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueMinimumLimit change.
    std::map<long, VehicleMotionManagementElectricAxleRow2TorqueMinimumLimitPropertyCb> m_torqueMinimumLimitCallbacks;
    // Mutex for m_torqueMinimumLimitCallbacks
    mutable std::shared_timed_mutex m_torqueMinimumLimitCallbacksMutex;
    // Next free unique identifier to subscribe for the RotationalSpeed change.
    std::atomic<long> m_rotationalSpeedChangedCallbackNextId {0};
    // Subscribed callbacks for the RotationalSpeed change.
    std::map<long, VehicleMotionManagementElectricAxleRow2RotationalSpeedPropertyCb> m_rotationalSpeedCallbacks;
    // Mutex for m_rotationalSpeedCallbacks
    mutable std::shared_timed_mutex m_rotationalSpeedCallbacksMutex;
    // Next free unique identifier to subscribe for the RotationalSpeedTarget change.
    std::atomic<long> m_rotationalSpeedTargetChangedCallbackNextId {0};
    // Subscribed callbacks for the RotationalSpeedTarget change.
    std::map<long, VehicleMotionManagementElectricAxleRow2RotationalSpeedTargetPropertyCb> m_rotationalSpeedTargetCallbacks;
    // Mutex for m_rotationalSpeedTargetCallbacks
    mutable std::shared_timed_mutex m_rotationalSpeedTargetCallbacksMutex;
    // Next free unique identifier to subscribe for the RotationalSpeedMaximumLimit change.
    std::atomic<long> m_rotationalSpeedMaximumLimitChangedCallbackNextId {0};
    // Subscribed callbacks for the RotationalSpeedMaximumLimit change.
    std::map<long, VehicleMotionManagementElectricAxleRow2RotationalSpeedMaximumLimitPropertyCb> m_rotationalSpeedMaximumLimitCallbacks;
    // Mutex for m_rotationalSpeedMaximumLimitCallbacks
    mutable std::shared_timed_mutex m_rotationalSpeedMaximumLimitCallbacksMutex;
    // Next free unique identifier to subscribe for the RotationalSpeedMinimumLimit change.
    std::atomic<long> m_rotationalSpeedMinimumLimitChangedCallbackNextId {0};
    // Subscribed callbacks for the RotationalSpeedMinimumLimit change.
    std::map<long, VehicleMotionManagementElectricAxleRow2RotationalSpeedMinimumLimitPropertyCb> m_rotationalSpeedMinimumLimitCallbacks;
    // Mutex for m_rotationalSpeedMinimumLimitCallbacks
    mutable std::shared_timed_mutex m_rotationalSpeedMinimumLimitCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
