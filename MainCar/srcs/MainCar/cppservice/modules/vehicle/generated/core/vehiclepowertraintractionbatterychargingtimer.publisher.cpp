

#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingtimer.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehiclePowertrainTractionBatteryChargingTimerPublisher::subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingTimerSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehiclePowertrainTractionBatteryChargingTimerSubscriber>(subscriber));
    }
}

void VehiclePowertrainTractionBatteryChargingTimerPublisher::unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingTimerSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehiclePowertrainTractionBatteryChargingTimerPublisher::subscribeToModeChanged(VehiclePowertrainTractionBatteryChargingTimerModePropertyCb callback)
{
    auto handleId = m_modeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_modeCallbacksMutex);
    m_modeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingTimerPublisher::unsubscribeFromModeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_modeCallbacksMutex);
    m_modeCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingTimerPublisher::publishModeChanged(Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum Mode) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onModeChanged(Mode);
    }
    std::shared_lock<std::shared_timed_mutex> modeCallbacksLock(m_modeCallbacksMutex);
    const auto modeCallbacks = m_modeCallbacks;
    modeCallbacksLock.unlock();
    for(const auto& callbackEntry: modeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Mode);
        }
    }
}

long VehiclePowertrainTractionBatteryChargingTimerPublisher::subscribeToTimeChanged(VehiclePowertrainTractionBatteryChargingTimerTimePropertyCb callback)
{
    auto handleId = m_timeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_timeCallbacksMutex);
    m_timeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingTimerPublisher::unsubscribeFromTimeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_timeCallbacksMutex);
    m_timeCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingTimerPublisher::publishTimeChanged(const std::string& Time) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTimeChanged(Time);
    }
    std::shared_lock<std::shared_timed_mutex> timeCallbacksLock(m_timeCallbacksMutex);
    const auto timeCallbacks = m_timeCallbacks;
    timeCallbacksLock.unlock();
    for(const auto& callbackEntry: timeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Time);
        }
    }
}

