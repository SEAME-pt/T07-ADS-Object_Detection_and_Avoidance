

#include "vehicle/generated/core/vehiclebodyhorn.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleBodyHornPublisher::subscribeToAllChanges(IVehicleBodyHornSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleBodyHornSubscriber>(subscriber));
    }
}

void VehicleBodyHornPublisher::unsubscribeFromAllChanges(IVehicleBodyHornSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleBodyHornPublisher::subscribeToIsActiveChanged(VehicleBodyHornIsActivePropertyCb callback)
{
    auto handleId = m_isActiveChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isActiveCallbacksMutex);
    m_isActiveCallbacks[handleId] = callback;
    return handleId;
}

void VehicleBodyHornPublisher::unsubscribeFromIsActiveChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isActiveCallbacksMutex);
    m_isActiveCallbacks.erase(handleId);
}

void VehicleBodyHornPublisher::publishIsActiveChanged(bool IsActive) const
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

