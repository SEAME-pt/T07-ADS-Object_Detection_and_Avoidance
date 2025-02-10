

#include "vehicle/generated/core/vehiclebodylightsbeamlow.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleBodyLightsBeamLowPublisher::subscribeToAllChanges(IVehicleBodyLightsBeamLowSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleBodyLightsBeamLowSubscriber>(subscriber));
    }
}

void VehicleBodyLightsBeamLowPublisher::unsubscribeFromAllChanges(IVehicleBodyLightsBeamLowSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleBodyLightsBeamLowPublisher::subscribeToIsOnChanged(VehicleBodyLightsBeamLowIsOnPropertyCb callback)
{
    auto handleId = m_isOnChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isOnCallbacksMutex);
    m_isOnCallbacks[handleId] = callback;
    return handleId;
}

void VehicleBodyLightsBeamLowPublisher::unsubscribeFromIsOnChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isOnCallbacksMutex);
    m_isOnCallbacks.erase(handleId);
}

void VehicleBodyLightsBeamLowPublisher::publishIsOnChanged(bool IsOn) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsOnChanged(IsOn);
    }
    std::shared_lock<std::shared_timed_mutex> isOnCallbacksLock(m_isOnCallbacksMutex);
    const auto isOnCallbacks = m_isOnCallbacks;
    isOnCallbacksLock.unlock();
    for(const auto& callbackEntry: isOnCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsOn);
        }
    }
}

long VehicleBodyLightsBeamLowPublisher::subscribeToIsDefectChanged(VehicleBodyLightsBeamLowIsDefectPropertyCb callback)
{
    auto handleId = m_isDefectChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectCallbacksMutex);
    m_isDefectCallbacks[handleId] = callback;
    return handleId;
}

void VehicleBodyLightsBeamLowPublisher::unsubscribeFromIsDefectChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectCallbacksMutex);
    m_isDefectCallbacks.erase(handleId);
}

void VehicleBodyLightsBeamLowPublisher::publishIsDefectChanged(bool IsDefect) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsDefectChanged(IsDefect);
    }
    std::shared_lock<std::shared_timed_mutex> isDefectCallbacksLock(m_isDefectCallbacksMutex);
    const auto isDefectCallbacks = m_isDefectCallbacks;
    isDefectCallbacksLock.unlock();
    for(const auto& callbackEntry: isDefectCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsDefect);
        }
    }
}

