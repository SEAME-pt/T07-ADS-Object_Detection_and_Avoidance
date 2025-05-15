

#include "vehicle/generated/core/vehiclepowertraintractionbatterybatteryconditioning.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToAllChanges(IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber>(subscriber));
    }
}

void VehiclePowertrainTractionBatteryBatteryConditioningPublisher::unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryBatteryConditioningSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToIsActiveChanged(VehiclePowertrainTractionBatteryBatteryConditioningIsActivePropertyCb callback)
{
    auto handleId = m_isActiveChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isActiveCallbacksMutex);
    m_isActiveCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryBatteryConditioningPublisher::unsubscribeFromIsActiveChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isActiveCallbacksMutex);
    m_isActiveCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryBatteryConditioningPublisher::publishIsActiveChanged(bool IsActive) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsActiveChanged(IsActive);
    }
    std::shared_lock<std::shared_timed_mutex> isActiveCallbacksLock(m_isActiveCallbacksMutex);
    const auto isActiveCallbacks = m_isActiveCallbacks;
    isActiveCallbacksLock.unlock();
    for(const auto& callbackEntry: isActiveCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsActive);
        }
    }
}

long VehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToIsOngoingChanged(VehiclePowertrainTractionBatteryBatteryConditioningIsOngoingPropertyCb callback)
{
    auto handleId = m_isOngoingChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isOngoingCallbacksMutex);
    m_isOngoingCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryBatteryConditioningPublisher::unsubscribeFromIsOngoingChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isOngoingCallbacksMutex);
    m_isOngoingCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryBatteryConditioningPublisher::publishIsOngoingChanged(bool IsOngoing) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsOngoingChanged(IsOngoing);
    }
    std::shared_lock<std::shared_timed_mutex> isOngoingCallbacksLock(m_isOngoingCallbacksMutex);
    const auto isOngoingCallbacks = m_isOngoingCallbacks;
    isOngoingCallbacksLock.unlock();
    for(const auto& callbackEntry: isOngoingCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsOngoing);
        }
    }
}

long VehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToRequestedModeChanged(VehiclePowertrainTractionBatteryBatteryConditioningRequestedModePropertyCb callback)
{
    auto handleId = m_requestedModeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_requestedModeCallbacksMutex);
    m_requestedModeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryBatteryConditioningPublisher::unsubscribeFromRequestedModeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_requestedModeCallbacksMutex);
    m_requestedModeCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryBatteryConditioningPublisher::publishRequestedModeChanged(Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum RequestedMode) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onRequestedModeChanged(RequestedMode);
    }
    std::shared_lock<std::shared_timed_mutex> requestedModeCallbacksLock(m_requestedModeCallbacksMutex);
    const auto requestedModeCallbacks = m_requestedModeCallbacks;
    requestedModeCallbacksLock.unlock();
    for(const auto& callbackEntry: requestedModeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(RequestedMode);
        }
    }
}

long VehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToStartTimeChanged(VehiclePowertrainTractionBatteryBatteryConditioningStartTimePropertyCb callback)
{
    auto handleId = m_startTimeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_startTimeCallbacksMutex);
    m_startTimeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryBatteryConditioningPublisher::unsubscribeFromStartTimeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_startTimeCallbacksMutex);
    m_startTimeCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryBatteryConditioningPublisher::publishStartTimeChanged(const std::string& StartTime) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onStartTimeChanged(StartTime);
    }
    std::shared_lock<std::shared_timed_mutex> startTimeCallbacksLock(m_startTimeCallbacksMutex);
    const auto startTimeCallbacks = m_startTimeCallbacks;
    startTimeCallbacksLock.unlock();
    for(const auto& callbackEntry: startTimeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(StartTime);
        }
    }
}

long VehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToTargetTimeChanged(VehiclePowertrainTractionBatteryBatteryConditioningTargetTimePropertyCb callback)
{
    auto handleId = m_targetTimeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_targetTimeCallbacksMutex);
    m_targetTimeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryBatteryConditioningPublisher::unsubscribeFromTargetTimeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_targetTimeCallbacksMutex);
    m_targetTimeCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryBatteryConditioningPublisher::publishTargetTimeChanged(const std::string& TargetTime) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTargetTimeChanged(TargetTime);
    }
    std::shared_lock<std::shared_timed_mutex> targetTimeCallbacksLock(m_targetTimeCallbacksMutex);
    const auto targetTimeCallbacks = m_targetTimeCallbacks;
    targetTimeCallbacksLock.unlock();
    for(const auto& callbackEntry: targetTimeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TargetTime);
        }
    }
}

long VehiclePowertrainTractionBatteryBatteryConditioningPublisher::subscribeToTargetTemperatureChanged(VehiclePowertrainTractionBatteryBatteryConditioningTargetTemperaturePropertyCb callback)
{
    auto handleId = m_targetTemperatureChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_targetTemperatureCallbacksMutex);
    m_targetTemperatureCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryBatteryConditioningPublisher::unsubscribeFromTargetTemperatureChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_targetTemperatureCallbacksMutex);
    m_targetTemperatureCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryBatteryConditioningPublisher::publishTargetTemperatureChanged(float TargetTemperature) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTargetTemperatureChanged(TargetTemperature);
    }
    std::shared_lock<std::shared_timed_mutex> targetTemperatureCallbacksLock(m_targetTemperatureCallbacksMutex);
    const auto targetTemperatureCallbacks = m_targetTemperatureCallbacks;
    targetTemperatureCallbacksLock.unlock();
    for(const auto& callbackEntry: targetTemperatureCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TargetTemperature);
        }
    }
}

