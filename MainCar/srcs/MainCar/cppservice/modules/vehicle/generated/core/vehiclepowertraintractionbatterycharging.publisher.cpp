

#include "vehicle/generated/core/vehiclepowertraintractionbatterycharging.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehiclePowertrainTractionBatteryChargingPublisher::subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehiclePowertrainTractionBatteryChargingSubscriber>(subscriber));
    }
}

void VehiclePowertrainTractionBatteryChargingPublisher::unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehiclePowertrainTractionBatteryChargingPublisher::subscribeToChargeLimitChanged(VehiclePowertrainTractionBatteryChargingChargeLimitPropertyCb callback)
{
    auto handleId = m_chargeLimitChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_chargeLimitCallbacksMutex);
    m_chargeLimitCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingPublisher::unsubscribeFromChargeLimitChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_chargeLimitCallbacksMutex);
    m_chargeLimitCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingPublisher::publishChargeLimitChanged(int32_t ChargeLimit) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onChargeLimitChanged(ChargeLimit);
    }
    std::shared_lock<std::shared_timed_mutex> chargeLimitCallbacksLock(m_chargeLimitCallbacksMutex);
    const auto chargeLimitCallbacks = m_chargeLimitCallbacks;
    chargeLimitCallbacksLock.unlock();
    for(const auto& callbackEntry: chargeLimitCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(ChargeLimit);
        }
    }
}

long VehiclePowertrainTractionBatteryChargingPublisher::subscribeToIsChargingChanged(VehiclePowertrainTractionBatteryChargingIsChargingPropertyCb callback)
{
    auto handleId = m_isChargingChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isChargingCallbacksMutex);
    m_isChargingCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingPublisher::unsubscribeFromIsChargingChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isChargingCallbacksMutex);
    m_isChargingCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingPublisher::publishIsChargingChanged(bool IsCharging) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsChargingChanged(IsCharging);
    }
    std::shared_lock<std::shared_timed_mutex> isChargingCallbacksLock(m_isChargingCallbacksMutex);
    const auto isChargingCallbacks = m_isChargingCallbacks;
    isChargingCallbacksLock.unlock();
    for(const auto& callbackEntry: isChargingCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsCharging);
        }
    }
}

long VehiclePowertrainTractionBatteryChargingPublisher::subscribeToIsDischargingChanged(VehiclePowertrainTractionBatteryChargingIsDischargingPropertyCb callback)
{
    auto handleId = m_isDischargingChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isDischargingCallbacksMutex);
    m_isDischargingCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingPublisher::unsubscribeFromIsDischargingChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isDischargingCallbacksMutex);
    m_isDischargingCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingPublisher::publishIsDischargingChanged(bool IsDischarging) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsDischargingChanged(IsDischarging);
    }
    std::shared_lock<std::shared_timed_mutex> isDischargingCallbacksLock(m_isDischargingCallbacksMutex);
    const auto isDischargingCallbacks = m_isDischargingCallbacks;
    isDischargingCallbacksLock.unlock();
    for(const auto& callbackEntry: isDischargingCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsDischarging);
        }
    }
}

long VehiclePowertrainTractionBatteryChargingPublisher::subscribeToStartStopChargingChanged(VehiclePowertrainTractionBatteryChargingStartStopChargingPropertyCb callback)
{
    auto handleId = m_startStopChargingChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_startStopChargingCallbacksMutex);
    m_startStopChargingCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingPublisher::unsubscribeFromStartStopChargingChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_startStopChargingCallbacksMutex);
    m_startStopChargingCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingPublisher::publishStartStopChargingChanged(Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum StartStopCharging) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onStartStopChargingChanged(StartStopCharging);
    }
    std::shared_lock<std::shared_timed_mutex> startStopChargingCallbacksLock(m_startStopChargingCallbacksMutex);
    const auto startStopChargingCallbacks = m_startStopChargingCallbacks;
    startStopChargingCallbacksLock.unlock();
    for(const auto& callbackEntry: startStopChargingCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(StartStopCharging);
        }
    }
}

long VehiclePowertrainTractionBatteryChargingPublisher::subscribeToAveragePowerChanged(VehiclePowertrainTractionBatteryChargingAveragePowerPropertyCb callback)
{
    auto handleId = m_averagePowerChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_averagePowerCallbacksMutex);
    m_averagePowerCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingPublisher::unsubscribeFromAveragePowerChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_averagePowerCallbacksMutex);
    m_averagePowerCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingPublisher::publishAveragePowerChanged(float AveragePower) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onAveragePowerChanged(AveragePower);
    }
    std::shared_lock<std::shared_timed_mutex> averagePowerCallbacksLock(m_averagePowerCallbacksMutex);
    const auto averagePowerCallbacks = m_averagePowerCallbacks;
    averagePowerCallbacksLock.unlock();
    for(const auto& callbackEntry: averagePowerCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(AveragePower);
        }
    }
}

long VehiclePowertrainTractionBatteryChargingPublisher::subscribeToMaxPowerChanged(VehiclePowertrainTractionBatteryChargingMaxPowerPropertyCb callback)
{
    auto handleId = m_maxPowerChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_maxPowerCallbacksMutex);
    m_maxPowerCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingPublisher::unsubscribeFromMaxPowerChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_maxPowerCallbacksMutex);
    m_maxPowerCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingPublisher::publishMaxPowerChanged(float MaxPower) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onMaxPowerChanged(MaxPower);
    }
    std::shared_lock<std::shared_timed_mutex> maxPowerCallbacksLock(m_maxPowerCallbacksMutex);
    const auto maxPowerCallbacks = m_maxPowerCallbacks;
    maxPowerCallbacksLock.unlock();
    for(const auto& callbackEntry: maxPowerCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(MaxPower);
        }
    }
}

long VehiclePowertrainTractionBatteryChargingPublisher::subscribeToEvseIdChanged(VehiclePowertrainTractionBatteryChargingEvseIdPropertyCb callback)
{
    auto handleId = m_evseIdChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_evseIdCallbacksMutex);
    m_evseIdCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingPublisher::unsubscribeFromEvseIdChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_evseIdCallbacksMutex);
    m_evseIdCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingPublisher::publishEvseIdChanged(const std::string& EvseId) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onEvseIdChanged(EvseId);
    }
    std::shared_lock<std::shared_timed_mutex> evseIdCallbacksLock(m_evseIdCallbacksMutex);
    const auto evseIdCallbacks = m_evseIdCallbacks;
    evseIdCallbacksLock.unlock();
    for(const auto& callbackEntry: evseIdCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(EvseId);
        }
    }
}

long VehiclePowertrainTractionBatteryChargingPublisher::subscribeToPowerLossChanged(VehiclePowertrainTractionBatteryChargingPowerLossPropertyCb callback)
{
    auto handleId = m_powerLossChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_powerLossCallbacksMutex);
    m_powerLossCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingPublisher::unsubscribeFromPowerLossChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_powerLossCallbacksMutex);
    m_powerLossCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingPublisher::publishPowerLossChanged(float PowerLoss) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onPowerLossChanged(PowerLoss);
    }
    std::shared_lock<std::shared_timed_mutex> powerLossCallbacksLock(m_powerLossCallbacksMutex);
    const auto powerLossCallbacks = m_powerLossCallbacks;
    powerLossCallbacksLock.unlock();
    for(const auto& callbackEntry: powerLossCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(PowerLoss);
        }
    }
}

long VehiclePowertrainTractionBatteryChargingPublisher::subscribeToTemperatureChanged(VehiclePowertrainTractionBatteryChargingTemperaturePropertyCb callback)
{
    auto handleId = m_temperatureChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_temperatureCallbacksMutex);
    m_temperatureCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingPublisher::unsubscribeFromTemperatureChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_temperatureCallbacksMutex);
    m_temperatureCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingPublisher::publishTemperatureChanged(float Temperature) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTemperatureChanged(Temperature);
    }
    std::shared_lock<std::shared_timed_mutex> temperatureCallbacksLock(m_temperatureCallbacksMutex);
    const auto temperatureCallbacks = m_temperatureCallbacks;
    temperatureCallbacksLock.unlock();
    for(const auto& callbackEntry: temperatureCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Temperature);
        }
    }
}

long VehiclePowertrainTractionBatteryChargingPublisher::subscribeToChargeRateChanged(VehiclePowertrainTractionBatteryChargingChargeRatePropertyCb callback)
{
    auto handleId = m_chargeRateChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_chargeRateCallbacksMutex);
    m_chargeRateCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingPublisher::unsubscribeFromChargeRateChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_chargeRateCallbacksMutex);
    m_chargeRateCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingPublisher::publishChargeRateChanged(float ChargeRate) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onChargeRateChanged(ChargeRate);
    }
    std::shared_lock<std::shared_timed_mutex> chargeRateCallbacksLock(m_chargeRateCallbacksMutex);
    const auto chargeRateCallbacks = m_chargeRateCallbacks;
    chargeRateCallbacksLock.unlock();
    for(const auto& callbackEntry: chargeRateCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(ChargeRate);
        }
    }
}

long VehiclePowertrainTractionBatteryChargingPublisher::subscribeToTimeToCompleteChanged(VehiclePowertrainTractionBatteryChargingTimeToCompletePropertyCb callback)
{
    auto handleId = m_timeToCompleteChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_timeToCompleteCallbacksMutex);
    m_timeToCompleteCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingPublisher::unsubscribeFromTimeToCompleteChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_timeToCompleteCallbacksMutex);
    m_timeToCompleteCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingPublisher::publishTimeToCompleteChanged(int32_t TimeToComplete) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTimeToCompleteChanged(TimeToComplete);
    }
    std::shared_lock<std::shared_timed_mutex> timeToCompleteCallbacksLock(m_timeToCompleteCallbacksMutex);
    const auto timeToCompleteCallbacks = m_timeToCompleteCallbacks;
    timeToCompleteCallbacksLock.unlock();
    for(const auto& callbackEntry: timeToCompleteCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TimeToComplete);
        }
    }
}

