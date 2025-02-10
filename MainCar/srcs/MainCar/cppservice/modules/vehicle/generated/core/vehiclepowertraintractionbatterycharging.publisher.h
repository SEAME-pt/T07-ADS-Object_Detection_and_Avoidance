#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertraintractionbatterycharging.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainTractionBatteryChargingPublisher.
 * Use this class to store clients of the Vehicle_Powertrain_TractionBattery_Charging and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingPublisher : public IVehiclePowertrainTractionBatteryChargingPublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToChargeLimitChanged
    */
    long subscribeToChargeLimitChanged(VehiclePowertrainTractionBatteryChargingChargeLimitPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToChargeLimitChanged
    */
    void unsubscribeFromChargeLimitChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToIsChargingChanged
    */
    long subscribeToIsChargingChanged(VehiclePowertrainTractionBatteryChargingIsChargingPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToIsChargingChanged
    */
    void unsubscribeFromIsChargingChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToIsDischargingChanged
    */
    long subscribeToIsDischargingChanged(VehiclePowertrainTractionBatteryChargingIsDischargingPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToIsDischargingChanged
    */
    void unsubscribeFromIsDischargingChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToStartStopChargingChanged
    */
    long subscribeToStartStopChargingChanged(VehiclePowertrainTractionBatteryChargingStartStopChargingPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToStartStopChargingChanged
    */
    void unsubscribeFromStartStopChargingChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToAveragePowerChanged
    */
    long subscribeToAveragePowerChanged(VehiclePowertrainTractionBatteryChargingAveragePowerPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToAveragePowerChanged
    */
    void unsubscribeFromAveragePowerChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToMaxPowerChanged
    */
    long subscribeToMaxPowerChanged(VehiclePowertrainTractionBatteryChargingMaxPowerPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToMaxPowerChanged
    */
    void unsubscribeFromMaxPowerChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToEvseIdChanged
    */
    long subscribeToEvseIdChanged(VehiclePowertrainTractionBatteryChargingEvseIdPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToEvseIdChanged
    */
    void unsubscribeFromEvseIdChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToPowerLossChanged
    */
    long subscribeToPowerLossChanged(VehiclePowertrainTractionBatteryChargingPowerLossPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToPowerLossChanged
    */
    void unsubscribeFromPowerLossChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToTemperatureChanged
    */
    long subscribeToTemperatureChanged(VehiclePowertrainTractionBatteryChargingTemperaturePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToTemperatureChanged
    */
    void unsubscribeFromTemperatureChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToChargeRateChanged
    */
    long subscribeToChargeRateChanged(VehiclePowertrainTractionBatteryChargingChargeRatePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToChargeRateChanged
    */
    void unsubscribeFromChargeRateChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToTimeToCompleteChanged
    */
    long subscribeToTimeToCompleteChanged(VehiclePowertrainTractionBatteryChargingTimeToCompletePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::subscribeToTimeToCompleteChanged
    */
    void unsubscribeFromTimeToCompleteChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::publishChargeLimitChanged
    */
    void publishChargeLimitChanged(int32_t ChargeLimit) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::publishIsChargingChanged
    */
    void publishIsChargingChanged(bool IsCharging) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::publishIsDischargingChanged
    */
    void publishIsDischargingChanged(bool IsDischarging) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::publishStartStopChargingChanged
    */
    void publishStartStopChargingChanged(Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum StartStopCharging) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::publishAveragePowerChanged
    */
    void publishAveragePowerChanged(float AveragePower) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::publishMaxPowerChanged
    */
    void publishMaxPowerChanged(float MaxPower) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::publishEvseIdChanged
    */
    void publishEvseIdChanged(const std::string& EvseId) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::publishPowerLossChanged
    */
    void publishPowerLossChanged(float PowerLoss) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::publishTemperatureChanged
    */
    void publishTemperatureChanged(float Temperature) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::publishChargeRateChanged
    */
    void publishChargeRateChanged(float ChargeRate) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingPublisher::publishTimeToCompleteChanged
    */
    void publishTimeToCompleteChanged(int32_t TimeToComplete) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_TractionBattery_Charging
    std::vector<std::reference_wrapper<IVehiclePowertrainTractionBatteryChargingSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the ChargeLimit change.
    std::atomic<long> m_chargeLimitChangedCallbackNextId {0};
    // Subscribed callbacks for the ChargeLimit change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargeLimitPropertyCb> m_chargeLimitCallbacks;
    // Mutex for m_chargeLimitCallbacks
    mutable std::shared_timed_mutex m_chargeLimitCallbacksMutex;
    // Next free unique identifier to subscribe for the IsCharging change.
    std::atomic<long> m_isChargingChangedCallbackNextId {0};
    // Subscribed callbacks for the IsCharging change.
    std::map<long, VehiclePowertrainTractionBatteryChargingIsChargingPropertyCb> m_isChargingCallbacks;
    // Mutex for m_isChargingCallbacks
    mutable std::shared_timed_mutex m_isChargingCallbacksMutex;
    // Next free unique identifier to subscribe for the IsDischarging change.
    std::atomic<long> m_isDischargingChangedCallbackNextId {0};
    // Subscribed callbacks for the IsDischarging change.
    std::map<long, VehiclePowertrainTractionBatteryChargingIsDischargingPropertyCb> m_isDischargingCallbacks;
    // Mutex for m_isDischargingCallbacks
    mutable std::shared_timed_mutex m_isDischargingCallbacksMutex;
    // Next free unique identifier to subscribe for the StartStopCharging change.
    std::atomic<long> m_startStopChargingChangedCallbackNextId {0};
    // Subscribed callbacks for the StartStopCharging change.
    std::map<long, VehiclePowertrainTractionBatteryChargingStartStopChargingPropertyCb> m_startStopChargingCallbacks;
    // Mutex for m_startStopChargingCallbacks
    mutable std::shared_timed_mutex m_startStopChargingCallbacksMutex;
    // Next free unique identifier to subscribe for the AveragePower change.
    std::atomic<long> m_averagePowerChangedCallbackNextId {0};
    // Subscribed callbacks for the AveragePower change.
    std::map<long, VehiclePowertrainTractionBatteryChargingAveragePowerPropertyCb> m_averagePowerCallbacks;
    // Mutex for m_averagePowerCallbacks
    mutable std::shared_timed_mutex m_averagePowerCallbacksMutex;
    // Next free unique identifier to subscribe for the MaxPower change.
    std::atomic<long> m_maxPowerChangedCallbackNextId {0};
    // Subscribed callbacks for the MaxPower change.
    std::map<long, VehiclePowertrainTractionBatteryChargingMaxPowerPropertyCb> m_maxPowerCallbacks;
    // Mutex for m_maxPowerCallbacks
    mutable std::shared_timed_mutex m_maxPowerCallbacksMutex;
    // Next free unique identifier to subscribe for the EvseId change.
    std::atomic<long> m_evseIdChangedCallbackNextId {0};
    // Subscribed callbacks for the EvseId change.
    std::map<long, VehiclePowertrainTractionBatteryChargingEvseIdPropertyCb> m_evseIdCallbacks;
    // Mutex for m_evseIdCallbacks
    mutable std::shared_timed_mutex m_evseIdCallbacksMutex;
    // Next free unique identifier to subscribe for the PowerLoss change.
    std::atomic<long> m_powerLossChangedCallbackNextId {0};
    // Subscribed callbacks for the PowerLoss change.
    std::map<long, VehiclePowertrainTractionBatteryChargingPowerLossPropertyCb> m_powerLossCallbacks;
    // Mutex for m_powerLossCallbacks
    mutable std::shared_timed_mutex m_powerLossCallbacksMutex;
    // Next free unique identifier to subscribe for the Temperature change.
    std::atomic<long> m_temperatureChangedCallbackNextId {0};
    // Subscribed callbacks for the Temperature change.
    std::map<long, VehiclePowertrainTractionBatteryChargingTemperaturePropertyCb> m_temperatureCallbacks;
    // Mutex for m_temperatureCallbacks
    mutable std::shared_timed_mutex m_temperatureCallbacksMutex;
    // Next free unique identifier to subscribe for the ChargeRate change.
    std::atomic<long> m_chargeRateChangedCallbackNextId {0};
    // Subscribed callbacks for the ChargeRate change.
    std::map<long, VehiclePowertrainTractionBatteryChargingChargeRatePropertyCb> m_chargeRateCallbacks;
    // Mutex for m_chargeRateCallbacks
    mutable std::shared_timed_mutex m_chargeRateCallbacksMutex;
    // Next free unique identifier to subscribe for the TimeToComplete change.
    std::atomic<long> m_timeToCompleteChangedCallbackNextId {0};
    // Subscribed callbacks for the TimeToComplete change.
    std::map<long, VehiclePowertrainTractionBatteryChargingTimeToCompletePropertyCb> m_timeToCompleteCallbacks;
    // Mutex for m_timeToCompleteCallbacks
    mutable std::shared_timed_mutex m_timeToCompleteCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
