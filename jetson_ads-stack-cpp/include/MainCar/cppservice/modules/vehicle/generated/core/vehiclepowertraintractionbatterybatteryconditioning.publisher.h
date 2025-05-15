#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertraintractionbatterybatteryconditioning.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainTractionBatteryBatteryConditioningPublisher.
 * Use this class to store clients of the Vehicle_Powertrain_TractionBattery_BatteryConditioning and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryBatteryConditioningPublisher : public IVehiclePowertrainTractionBatteryBatteryConditioningPublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToIsActiveChanged
    */
    long subscribeToIsActiveChanged(VehiclePowertrainTractionBatteryBatteryConditioningIsActivePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToIsActiveChanged
    */
    void unsubscribeFromIsActiveChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToIsOngoingChanged
    */
    long subscribeToIsOngoingChanged(VehiclePowertrainTractionBatteryBatteryConditioningIsOngoingPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToIsOngoingChanged
    */
    void unsubscribeFromIsOngoingChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToRequestedModeChanged
    */
    long subscribeToRequestedModeChanged(VehiclePowertrainTractionBatteryBatteryConditioningRequestedModePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToRequestedModeChanged
    */
    void unsubscribeFromRequestedModeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToStartTimeChanged
    */
    long subscribeToStartTimeChanged(VehiclePowertrainTractionBatteryBatteryConditioningStartTimePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToStartTimeChanged
    */
    void unsubscribeFromStartTimeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToTargetTimeChanged
    */
    long subscribeToTargetTimeChanged(VehiclePowertrainTractionBatteryBatteryConditioningTargetTimePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToTargetTimeChanged
    */
    void unsubscribeFromTargetTimeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToTargetTemperatureChanged
    */
    long subscribeToTargetTemperatureChanged(VehiclePowertrainTractionBatteryBatteryConditioningTargetTemperaturePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToTargetTemperatureChanged
    */
    void unsubscribeFromTargetTemperatureChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::publishIsActiveChanged
    */
    void publishIsActiveChanged(bool IsActive) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::publishIsOngoingChanged
    */
    void publishIsOngoingChanged(bool IsOngoing) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::publishRequestedModeChanged
    */
    void publishRequestedModeChanged(Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum RequestedMode) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::publishStartTimeChanged
    */
    void publishStartTimeChanged(const std::string& StartTime) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::publishTargetTimeChanged
    */
    void publishTargetTimeChanged(const std::string& TargetTime) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryBatteryConditioningPublisher::publishTargetTemperatureChanged
    */
    void publishTargetTemperatureChanged(float TargetTemperature) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_TractionBattery_BatteryConditioning
    std::vector<std::reference_wrapper<IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the IsActive change.
    std::atomic<long> m_isActiveChangedCallbackNextId {0};
    // Subscribed callbacks for the IsActive change.
    std::map<long, VehiclePowertrainTractionBatteryBatteryConditioningIsActivePropertyCb> m_isActiveCallbacks;
    // Mutex for m_isActiveCallbacks
    mutable std::shared_timed_mutex m_isActiveCallbacksMutex;
    // Next free unique identifier to subscribe for the IsOngoing change.
    std::atomic<long> m_isOngoingChangedCallbackNextId {0};
    // Subscribed callbacks for the IsOngoing change.
    std::map<long, VehiclePowertrainTractionBatteryBatteryConditioningIsOngoingPropertyCb> m_isOngoingCallbacks;
    // Mutex for m_isOngoingCallbacks
    mutable std::shared_timed_mutex m_isOngoingCallbacksMutex;
    // Next free unique identifier to subscribe for the RequestedMode change.
    std::atomic<long> m_requestedModeChangedCallbackNextId {0};
    // Subscribed callbacks for the RequestedMode change.
    std::map<long, VehiclePowertrainTractionBatteryBatteryConditioningRequestedModePropertyCb> m_requestedModeCallbacks;
    // Mutex for m_requestedModeCallbacks
    mutable std::shared_timed_mutex m_requestedModeCallbacksMutex;
    // Next free unique identifier to subscribe for the StartTime change.
    std::atomic<long> m_startTimeChangedCallbackNextId {0};
    // Subscribed callbacks for the StartTime change.
    std::map<long, VehiclePowertrainTractionBatteryBatteryConditioningStartTimePropertyCb> m_startTimeCallbacks;
    // Mutex for m_startTimeCallbacks
    mutable std::shared_timed_mutex m_startTimeCallbacksMutex;
    // Next free unique identifier to subscribe for the TargetTime change.
    std::atomic<long> m_targetTimeChangedCallbackNextId {0};
    // Subscribed callbacks for the TargetTime change.
    std::map<long, VehiclePowertrainTractionBatteryBatteryConditioningTargetTimePropertyCb> m_targetTimeCallbacks;
    // Mutex for m_targetTimeCallbacks
    mutable std::shared_timed_mutex m_targetTimeCallbacksMutex;
    // Next free unique identifier to subscribe for the TargetTemperature change.
    std::atomic<long> m_targetTemperatureChangedCallbackNextId {0};
    // Subscribed callbacks for the TargetTemperature change.
    std::map<long, VehiclePowertrainTractionBatteryBatteryConditioningTargetTemperaturePropertyCb> m_targetTemperatureCallbacks;
    // Mutex for m_targetTemperatureCallbacks
    mutable std::shared_timed_mutex m_targetTemperatureCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
