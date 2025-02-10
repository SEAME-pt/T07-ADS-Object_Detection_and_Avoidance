

#include "vehicle/generated/core/vehiclebody.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleBodyPublisher::subscribeToAllChanges(IVehicleBodySubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleBodySubscriber>(subscriber));
    }
}

void VehicleBodyPublisher::unsubscribeFromAllChanges(IVehicleBodySubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleBodyPublisher::subscribeToPowerOptimizeLevelChanged(VehicleBodyPowerOptimizeLevelPropertyCb callback)
{
    auto handleId = m_powerOptimizeLevelChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_powerOptimizeLevelCallbacksMutex);
    m_powerOptimizeLevelCallbacks[handleId] = callback;
    return handleId;
}

void VehicleBodyPublisher::unsubscribeFromPowerOptimizeLevelChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_powerOptimizeLevelCallbacksMutex);
    m_powerOptimizeLevelCallbacks.erase(handleId);
}

void VehicleBodyPublisher::publishPowerOptimizeLevelChanged(int32_t PowerOptimizeLevel) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onPowerOptimizeLevelChanged(PowerOptimizeLevel);
    }
    std::shared_lock<std::shared_timed_mutex> powerOptimizeLevelCallbacksLock(m_powerOptimizeLevelCallbacksMutex);
    const auto powerOptimizeLevelCallbacks = m_powerOptimizeLevelCallbacks;
    powerOptimizeLevelCallbacksLock.unlock();
    for(const auto& callbackEntry: powerOptimizeLevelCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(PowerOptimizeLevel);
        }
    }
}

long VehicleBodyPublisher::subscribeToIsAutoPowerOptimizeChanged(VehicleBodyIsAutoPowerOptimizePropertyCb callback)
{
    auto handleId = m_isAutoPowerOptimizeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isAutoPowerOptimizeCallbacksMutex);
    m_isAutoPowerOptimizeCallbacks[handleId] = callback;
    return handleId;
}

void VehicleBodyPublisher::unsubscribeFromIsAutoPowerOptimizeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isAutoPowerOptimizeCallbacksMutex);
    m_isAutoPowerOptimizeCallbacks.erase(handleId);
}

void VehicleBodyPublisher::publishIsAutoPowerOptimizeChanged(bool IsAutoPowerOptimize) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsAutoPowerOptimizeChanged(IsAutoPowerOptimize);
    }
    std::shared_lock<std::shared_timed_mutex> isAutoPowerOptimizeCallbacksLock(m_isAutoPowerOptimizeCallbacksMutex);
    const auto isAutoPowerOptimizeCallbacks = m_isAutoPowerOptimizeCallbacks;
    isAutoPowerOptimizeCallbacksLock.unlock();
    for(const auto& callbackEntry: isAutoPowerOptimizeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsAutoPowerOptimize);
        }
    }
}

