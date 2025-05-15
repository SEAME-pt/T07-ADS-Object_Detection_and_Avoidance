#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclemotionmanagementelectricaxlerow1.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleMotionManagementElectricAxleRow1Publisher.
 * Use this class to store clients of the Vehicle_MotionManagement_ElectricAxle_Row1 and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleMotionManagementElectricAxleRow1Publisher : public IVehicleMotionManagementElectricAxleRow1Publisher
{
public:
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleMotionManagementElectricAxleRow1Subscriber& subscriber) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleMotionManagementElectricAxleRow1Subscriber& subscriber) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueChanged
    */
    long subscribeToTorqueChanged(VehicleMotionManagementElectricAxleRow1TorquePropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueChanged
    */
    void unsubscribeFromTorqueChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueTargetChanged
    */
    long subscribeToTorqueTargetChanged(VehicleMotionManagementElectricAxleRow1TorqueTargetPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueTargetChanged
    */
    void unsubscribeFromTorqueTargetChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueMaximumChanged
    */
    long subscribeToTorqueMaximumChanged(VehicleMotionManagementElectricAxleRow1TorqueMaximumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueMaximumChanged
    */
    void unsubscribeFromTorqueMaximumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueMinimumChanged
    */
    long subscribeToTorqueMinimumChanged(VehicleMotionManagementElectricAxleRow1TorqueMinimumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueMinimumChanged
    */
    void unsubscribeFromTorqueMinimumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueLongTermMaximumChanged
    */
    long subscribeToTorqueLongTermMaximumChanged(VehicleMotionManagementElectricAxleRow1TorqueLongTermMaximumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueLongTermMaximumChanged
    */
    void unsubscribeFromTorqueLongTermMaximumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueLongTermMinimumChanged
    */
    long subscribeToTorqueLongTermMinimumChanged(VehicleMotionManagementElectricAxleRow1TorqueLongTermMinimumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueLongTermMinimumChanged
    */
    void unsubscribeFromTorqueLongTermMinimumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueShortTermMaximumChanged
    */
    long subscribeToTorqueShortTermMaximumChanged(VehicleMotionManagementElectricAxleRow1TorqueShortTermMaximumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueShortTermMaximumChanged
    */
    void unsubscribeFromTorqueShortTermMaximumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueShortTermMinimumChanged
    */
    long subscribeToTorqueShortTermMinimumChanged(VehicleMotionManagementElectricAxleRow1TorqueShortTermMinimumPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueShortTermMinimumChanged
    */
    void unsubscribeFromTorqueShortTermMinimumChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueMaximumLimitChanged
    */
    long subscribeToTorqueMaximumLimitChanged(VehicleMotionManagementElectricAxleRow1TorqueMaximumLimitPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueMaximumLimitChanged
    */
    void unsubscribeFromTorqueMaximumLimitChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueMinimumLimitChanged
    */
    long subscribeToTorqueMinimumLimitChanged(VehicleMotionManagementElectricAxleRow1TorqueMinimumLimitPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToTorqueMinimumLimitChanged
    */
    void unsubscribeFromTorqueMinimumLimitChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToRotationalSpeedChanged
    */
    long subscribeToRotationalSpeedChanged(VehicleMotionManagementElectricAxleRow1RotationalSpeedPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToRotationalSpeedChanged
    */
    void unsubscribeFromRotationalSpeedChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToRotationalSpeedTargetChanged
    */
    long subscribeToRotationalSpeedTargetChanged(VehicleMotionManagementElectricAxleRow1RotationalSpeedTargetPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToRotationalSpeedTargetChanged
    */
    void unsubscribeFromRotationalSpeedTargetChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToRotationalSpeedMaximumLimitChanged
    */
    long subscribeToRotationalSpeedMaximumLimitChanged(VehicleMotionManagementElectricAxleRow1RotationalSpeedMaximumLimitPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToRotationalSpeedMaximumLimitChanged
    */
    void unsubscribeFromRotationalSpeedMaximumLimitChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToRotationalSpeedMinimumLimitChanged
    */
    long subscribeToRotationalSpeedMinimumLimitChanged(VehicleMotionManagementElectricAxleRow1RotationalSpeedMinimumLimitPropertyCb callback) override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::subscribeToRotationalSpeedMinimumLimitChanged
    */
    void unsubscribeFromRotationalSpeedMinimumLimitChanged(long handleId) override;

    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::publishTorqueChanged
    */
    void publishTorqueChanged(int32_t Torque) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::publishTorqueTargetChanged
    */
    void publishTorqueTargetChanged(int32_t TorqueTarget) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::publishTorqueMaximumChanged
    */
    void publishTorqueMaximumChanged(int32_t TorqueMaximum) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::publishTorqueMinimumChanged
    */
    void publishTorqueMinimumChanged(int32_t TorqueMinimum) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::publishTorqueLongTermMaximumChanged
    */
    void publishTorqueLongTermMaximumChanged(int32_t TorqueLongTermMaximum) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::publishTorqueLongTermMinimumChanged
    */
    void publishTorqueLongTermMinimumChanged(int32_t TorqueLongTermMinimum) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::publishTorqueShortTermMaximumChanged
    */
    void publishTorqueShortTermMaximumChanged(int32_t TorqueShortTermMaximum) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::publishTorqueShortTermMinimumChanged
    */
    void publishTorqueShortTermMinimumChanged(int32_t TorqueShortTermMinimum) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::publishTorqueMaximumLimitChanged
    */
    void publishTorqueMaximumLimitChanged(int32_t TorqueMaximumLimit) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::publishTorqueMinimumLimitChanged
    */
    void publishTorqueMinimumLimitChanged(int32_t TorqueMinimumLimit) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::publishRotationalSpeedChanged
    */
    void publishRotationalSpeedChanged(int32_t RotationalSpeed) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::publishRotationalSpeedTargetChanged
    */
    void publishRotationalSpeedTargetChanged(int32_t RotationalSpeedTarget) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::publishRotationalSpeedMaximumLimitChanged
    */
    void publishRotationalSpeedMaximumLimitChanged(int32_t RotationalSpeedMaximumLimit) const override;
    /**
    * Implementation of IVehicleMotionManagementElectricAxleRow1Publisher::publishRotationalSpeedMinimumLimitChanged
    */
    void publishRotationalSpeedMinimumLimitChanged(int32_t RotationalSpeedMinimumLimit) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_MotionManagement_ElectricAxle_Row1
    std::vector<std::reference_wrapper<IVehicleMotionManagementElectricAxleRow1Subscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Torque change.
    std::atomic<long> m_torqueChangedCallbackNextId {0};
    // Subscribed callbacks for the Torque change.
    std::map<long, VehicleMotionManagementElectricAxleRow1TorquePropertyCb> m_torqueCallbacks;
    // Mutex for m_torqueCallbacks
    mutable std::shared_timed_mutex m_torqueCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueTarget change.
    std::atomic<long> m_torqueTargetChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueTarget change.
    std::map<long, VehicleMotionManagementElectricAxleRow1TorqueTargetPropertyCb> m_torqueTargetCallbacks;
    // Mutex for m_torqueTargetCallbacks
    mutable std::shared_timed_mutex m_torqueTargetCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueMaximum change.
    std::atomic<long> m_torqueMaximumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueMaximum change.
    std::map<long, VehicleMotionManagementElectricAxleRow1TorqueMaximumPropertyCb> m_torqueMaximumCallbacks;
    // Mutex for m_torqueMaximumCallbacks
    mutable std::shared_timed_mutex m_torqueMaximumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueMinimum change.
    std::atomic<long> m_torqueMinimumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueMinimum change.
    std::map<long, VehicleMotionManagementElectricAxleRow1TorqueMinimumPropertyCb> m_torqueMinimumCallbacks;
    // Mutex for m_torqueMinimumCallbacks
    mutable std::shared_timed_mutex m_torqueMinimumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueLongTermMaximum change.
    std::atomic<long> m_torqueLongTermMaximumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueLongTermMaximum change.
    std::map<long, VehicleMotionManagementElectricAxleRow1TorqueLongTermMaximumPropertyCb> m_torqueLongTermMaximumCallbacks;
    // Mutex for m_torqueLongTermMaximumCallbacks
    mutable std::shared_timed_mutex m_torqueLongTermMaximumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueLongTermMinimum change.
    std::atomic<long> m_torqueLongTermMinimumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueLongTermMinimum change.
    std::map<long, VehicleMotionManagementElectricAxleRow1TorqueLongTermMinimumPropertyCb> m_torqueLongTermMinimumCallbacks;
    // Mutex for m_torqueLongTermMinimumCallbacks
    mutable std::shared_timed_mutex m_torqueLongTermMinimumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueShortTermMaximum change.
    std::atomic<long> m_torqueShortTermMaximumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueShortTermMaximum change.
    std::map<long, VehicleMotionManagementElectricAxleRow1TorqueShortTermMaximumPropertyCb> m_torqueShortTermMaximumCallbacks;
    // Mutex for m_torqueShortTermMaximumCallbacks
    mutable std::shared_timed_mutex m_torqueShortTermMaximumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueShortTermMinimum change.
    std::atomic<long> m_torqueShortTermMinimumChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueShortTermMinimum change.
    std::map<long, VehicleMotionManagementElectricAxleRow1TorqueShortTermMinimumPropertyCb> m_torqueShortTermMinimumCallbacks;
    // Mutex for m_torqueShortTermMinimumCallbacks
    mutable std::shared_timed_mutex m_torqueShortTermMinimumCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueMaximumLimit change.
    std::atomic<long> m_torqueMaximumLimitChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueMaximumLimit change.
    std::map<long, VehicleMotionManagementElectricAxleRow1TorqueMaximumLimitPropertyCb> m_torqueMaximumLimitCallbacks;
    // Mutex for m_torqueMaximumLimitCallbacks
    mutable std::shared_timed_mutex m_torqueMaximumLimitCallbacksMutex;
    // Next free unique identifier to subscribe for the TorqueMinimumLimit change.
    std::atomic<long> m_torqueMinimumLimitChangedCallbackNextId {0};
    // Subscribed callbacks for the TorqueMinimumLimit change.
    std::map<long, VehicleMotionManagementElectricAxleRow1TorqueMinimumLimitPropertyCb> m_torqueMinimumLimitCallbacks;
    // Mutex for m_torqueMinimumLimitCallbacks
    mutable std::shared_timed_mutex m_torqueMinimumLimitCallbacksMutex;
    // Next free unique identifier to subscribe for the RotationalSpeed change.
    std::atomic<long> m_rotationalSpeedChangedCallbackNextId {0};
    // Subscribed callbacks for the RotationalSpeed change.
    std::map<long, VehicleMotionManagementElectricAxleRow1RotationalSpeedPropertyCb> m_rotationalSpeedCallbacks;
    // Mutex for m_rotationalSpeedCallbacks
    mutable std::shared_timed_mutex m_rotationalSpeedCallbacksMutex;
    // Next free unique identifier to subscribe for the RotationalSpeedTarget change.
    std::atomic<long> m_rotationalSpeedTargetChangedCallbackNextId {0};
    // Subscribed callbacks for the RotationalSpeedTarget change.
    std::map<long, VehicleMotionManagementElectricAxleRow1RotationalSpeedTargetPropertyCb> m_rotationalSpeedTargetCallbacks;
    // Mutex for m_rotationalSpeedTargetCallbacks
    mutable std::shared_timed_mutex m_rotationalSpeedTargetCallbacksMutex;
    // Next free unique identifier to subscribe for the RotationalSpeedMaximumLimit change.
    std::atomic<long> m_rotationalSpeedMaximumLimitChangedCallbackNextId {0};
    // Subscribed callbacks for the RotationalSpeedMaximumLimit change.
    std::map<long, VehicleMotionManagementElectricAxleRow1RotationalSpeedMaximumLimitPropertyCb> m_rotationalSpeedMaximumLimitCallbacks;
    // Mutex for m_rotationalSpeedMaximumLimitCallbacks
    mutable std::shared_timed_mutex m_rotationalSpeedMaximumLimitCallbacksMutex;
    // Next free unique identifier to subscribe for the RotationalSpeedMinimumLimit change.
    std::atomic<long> m_rotationalSpeedMinimumLimitChangedCallbackNextId {0};
    // Subscribed callbacks for the RotationalSpeedMinimumLimit change.
    std::map<long, VehicleMotionManagementElectricAxleRow1RotationalSpeedMinimumLimitPropertyCb> m_rotationalSpeedMinimumLimitCallbacks;
    // Mutex for m_rotationalSpeedMinimumLimitCallbacks
    mutable std::shared_timed_mutex m_rotationalSpeedMinimumLimitCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
