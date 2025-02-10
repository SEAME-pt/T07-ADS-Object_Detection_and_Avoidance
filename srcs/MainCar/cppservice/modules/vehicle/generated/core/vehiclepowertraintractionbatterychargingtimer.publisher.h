#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertraintractionbatterychargingtimer.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainTractionBatteryChargingTimerPublisher.
 * Use this class to store clients of the Vehicle_Powertrain_TractionBattery_Charging_Timer and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingTimerPublisher : public IVehiclePowertrainTractionBatteryChargingTimerPublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingTimerPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingTimerSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingTimerPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingTimerSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingTimerPublisher::subscribeToModeChanged
    */
    long subscribeToModeChanged(VehiclePowertrainTractionBatteryChargingTimerModePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingTimerPublisher::subscribeToModeChanged
    */
    void unsubscribeFromModeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingTimerPublisher::subscribeToTimeChanged
    */
    long subscribeToTimeChanged(VehiclePowertrainTractionBatteryChargingTimerTimePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingTimerPublisher::subscribeToTimeChanged
    */
    void unsubscribeFromTimeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingTimerPublisher::publishModeChanged
    */
    void publishModeChanged(Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum Mode) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingTimerPublisher::publishTimeChanged
    */
    void publishTimeChanged(const std::string& Time) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_TractionBattery_Charging_Timer
    std::vector<std::reference_wrapper<IVehiclePowertrainTractionBatteryChargingTimerSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Mode change.
    std::atomic<long> m_modeChangedCallbackNextId {0};
    // Subscribed callbacks for the Mode change.
    std::map<long, VehiclePowertrainTractionBatteryChargingTimerModePropertyCb> m_modeCallbacks;
    // Mutex for m_modeCallbacks
    mutable std::shared_timed_mutex m_modeCallbacksMutex;
    // Next free unique identifier to subscribe for the Time change.
    std::atomic<long> m_timeChangedCallbackNextId {0};
    // Subscribed callbacks for the Time change.
    std::map<long, VehiclePowertrainTractionBatteryChargingTimerTimePropertyCb> m_timeCallbacks;
    // Mutex for m_timeCallbacks
    mutable std::shared_timed_mutex m_timeCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
