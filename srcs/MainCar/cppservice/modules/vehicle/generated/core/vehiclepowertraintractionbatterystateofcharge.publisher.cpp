

#include "vehicle/generated/core/vehiclepowertraintractionbatterystateofcharge.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehiclePowertrainTractionBatteryStateOfChargePublisher::subscribeToAllChanges(IVehiclePowertrainTractionBatteryStateOfChargeSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehiclePowertrainTractionBatteryStateOfChargeSubscriber>(subscriber));
    }
}

void VehiclePowertrainTractionBatteryStateOfChargePublisher::unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryStateOfChargeSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehiclePowertrainTractionBatteryStateOfChargePublisher::subscribeToCurrentChanged(VehiclePowertrainTractionBatteryStateOfChargeCurrentPropertyCb callback)
{
    auto handleId = m_currentChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_currentCallbacksMutex);
    m_currentCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryStateOfChargePublisher::unsubscribeFromCurrentChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_currentCallbacksMutex);
    m_currentCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryStateOfChargePublisher::publishCurrentChanged(float Current) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onCurrentChanged(Current);
    }
    std::shared_lock<std::shared_timed_mutex> currentCallbacksLock(m_currentCallbacksMutex);
    const auto currentCallbacks = m_currentCallbacks;
    currentCallbacksLock.unlock();
    for(const auto& callbackEntry: currentCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Current);
        }
    }
}

long VehiclePowertrainTractionBatteryStateOfChargePublisher::subscribeToDisplayedChanged(VehiclePowertrainTractionBatteryStateOfChargeDisplayedPropertyCb callback)
{
    auto handleId = m_displayedChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_displayedCallbacksMutex);
    m_displayedCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryStateOfChargePublisher::unsubscribeFromDisplayedChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_displayedCallbacksMutex);
    m_displayedCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryStateOfChargePublisher::publishDisplayedChanged(float Displayed) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onDisplayedChanged(Displayed);
    }
    std::shared_lock<std::shared_timed_mutex> displayedCallbacksLock(m_displayedCallbacksMutex);
    const auto displayedCallbacks = m_displayedCallbacks;
    displayedCallbacksLock.unlock();
    for(const auto& callbackEntry: displayedCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Displayed);
        }
    }
}

long VehiclePowertrainTractionBatteryStateOfChargePublisher::subscribeToCurrentEnergyChanged(VehiclePowertrainTractionBatteryStateOfChargeCurrentEnergyPropertyCb callback)
{
    auto handleId = m_currentEnergyChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_currentEnergyCallbacksMutex);
    m_currentEnergyCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryStateOfChargePublisher::unsubscribeFromCurrentEnergyChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_currentEnergyCallbacksMutex);
    m_currentEnergyCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryStateOfChargePublisher::publishCurrentEnergyChanged(float CurrentEnergy) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onCurrentEnergyChanged(CurrentEnergy);
    }
    std::shared_lock<std::shared_timed_mutex> currentEnergyCallbacksLock(m_currentEnergyCallbacksMutex);
    const auto currentEnergyCallbacks = m_currentEnergyCallbacks;
    currentEnergyCallbacksLock.unlock();
    for(const auto& callbackEntry: currentEnergyCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(CurrentEnergy);
        }
    }
}

