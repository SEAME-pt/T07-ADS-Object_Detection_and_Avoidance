

#include "vehicle/generated/core/vehicleconnectivity.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleConnectivityPublisher::subscribeToAllChanges(IVehicleConnectivitySubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleConnectivitySubscriber>(subscriber));
    }
}

void VehicleConnectivityPublisher::unsubscribeFromAllChanges(IVehicleConnectivitySubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleConnectivityPublisher::subscribeToIsConnectivityAvailableChanged(VehicleConnectivityIsConnectivityAvailablePropertyCb callback)
{
    auto handleId = m_isConnectivityAvailableChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isConnectivityAvailableCallbacksMutex);
    m_isConnectivityAvailableCallbacks[handleId] = callback;
    return handleId;
}

void VehicleConnectivityPublisher::unsubscribeFromIsConnectivityAvailableChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isConnectivityAvailableCallbacksMutex);
    m_isConnectivityAvailableCallbacks.erase(handleId);
}

void VehicleConnectivityPublisher::publishIsConnectivityAvailableChanged(bool IsConnectivityAvailable) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsConnectivityAvailableChanged(IsConnectivityAvailable);
    }
    std::shared_lock<std::shared_timed_mutex> isConnectivityAvailableCallbacksLock(m_isConnectivityAvailableCallbacksMutex);
    const auto isConnectivityAvailableCallbacks = m_isConnectivityAvailableCallbacks;
    isConnectivityAvailableCallbacksLock.unlock();
    for(const auto& callbackEntry: isConnectivityAvailableCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsConnectivityAvailable);
        }
    }
}

