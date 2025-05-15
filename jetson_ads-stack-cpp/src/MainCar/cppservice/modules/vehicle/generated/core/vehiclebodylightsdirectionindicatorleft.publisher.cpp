

#include "vehicle/generated/core/vehiclebodylightsdirectionindicatorleft.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleBodyLightsDirectionIndicatorLeftPublisher::subscribeToAllChanges(IVehicleBodyLightsDirectionIndicatorLeftSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleBodyLightsDirectionIndicatorLeftSubscriber>(subscriber));
    }
}

void VehicleBodyLightsDirectionIndicatorLeftPublisher::unsubscribeFromAllChanges(IVehicleBodyLightsDirectionIndicatorLeftSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleBodyLightsDirectionIndicatorLeftPublisher::subscribeToIsSignalingChanged(VehicleBodyLightsDirectionIndicatorLeftIsSignalingPropertyCb callback)
{
    auto handleId = m_isSignalingChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isSignalingCallbacksMutex);
    m_isSignalingCallbacks[handleId] = callback;
    return handleId;
}

void VehicleBodyLightsDirectionIndicatorLeftPublisher::unsubscribeFromIsSignalingChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isSignalingCallbacksMutex);
    m_isSignalingCallbacks.erase(handleId);
}

void VehicleBodyLightsDirectionIndicatorLeftPublisher::publishIsSignalingChanged(bool IsSignaling) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsSignalingChanged(IsSignaling);
    }
    std::shared_lock<std::shared_timed_mutex> isSignalingCallbacksLock(m_isSignalingCallbacksMutex);
    const auto isSignalingCallbacks = m_isSignalingCallbacks;
    isSignalingCallbacksLock.unlock();
    for(const auto& callbackEntry: isSignalingCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsSignaling);
        }
    }
}

long VehicleBodyLightsDirectionIndicatorLeftPublisher::subscribeToIsDefectChanged(VehicleBodyLightsDirectionIndicatorLeftIsDefectPropertyCb callback)
{
    auto handleId = m_isDefectChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectCallbacksMutex);
    m_isDefectCallbacks[handleId] = callback;
    return handleId;
}

void VehicleBodyLightsDirectionIndicatorLeftPublisher::unsubscribeFromIsDefectChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectCallbacksMutex);
    m_isDefectCallbacks.erase(handleId);
}

void VehicleBodyLightsDirectionIndicatorLeftPublisher::publishIsDefectChanged(bool IsDefect) const
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

