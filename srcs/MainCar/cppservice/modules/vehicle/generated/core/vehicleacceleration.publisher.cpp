

#include "vehicle/generated/core/vehicleacceleration.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleAccelerationPublisher::subscribeToAllChanges(IVehicleAccelerationSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleAccelerationSubscriber>(subscriber));
    }
}

void VehicleAccelerationPublisher::unsubscribeFromAllChanges(IVehicleAccelerationSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleAccelerationPublisher::subscribeToLongitudinalChanged(VehicleAccelerationLongitudinalPropertyCb callback)
{
    auto handleId = m_longitudinalChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_longitudinalCallbacksMutex);
    m_longitudinalCallbacks[handleId] = callback;
    return handleId;
}

void VehicleAccelerationPublisher::unsubscribeFromLongitudinalChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_longitudinalCallbacksMutex);
    m_longitudinalCallbacks.erase(handleId);
}

void VehicleAccelerationPublisher::publishLongitudinalChanged(float Longitudinal) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onLongitudinalChanged(Longitudinal);
    }
    std::shared_lock<std::shared_timed_mutex> longitudinalCallbacksLock(m_longitudinalCallbacksMutex);
    const auto longitudinalCallbacks = m_longitudinalCallbacks;
    longitudinalCallbacksLock.unlock();
    for(const auto& callbackEntry: longitudinalCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Longitudinal);
        }
    }
}

long VehicleAccelerationPublisher::subscribeToLateralChanged(VehicleAccelerationLateralPropertyCb callback)
{
    auto handleId = m_lateralChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_lateralCallbacksMutex);
    m_lateralCallbacks[handleId] = callback;
    return handleId;
}

void VehicleAccelerationPublisher::unsubscribeFromLateralChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_lateralCallbacksMutex);
    m_lateralCallbacks.erase(handleId);
}

void VehicleAccelerationPublisher::publishLateralChanged(float Lateral) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onLateralChanged(Lateral);
    }
    std::shared_lock<std::shared_timed_mutex> lateralCallbacksLock(m_lateralCallbacksMutex);
    const auto lateralCallbacks = m_lateralCallbacks;
    lateralCallbacksLock.unlock();
    for(const auto& callbackEntry: lateralCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Lateral);
        }
    }
}

long VehicleAccelerationPublisher::subscribeToVerticalChanged(VehicleAccelerationVerticalPropertyCb callback)
{
    auto handleId = m_verticalChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_verticalCallbacksMutex);
    m_verticalCallbacks[handleId] = callback;
    return handleId;
}

void VehicleAccelerationPublisher::unsubscribeFromVerticalChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_verticalCallbacksMutex);
    m_verticalCallbacks.erase(handleId);
}

void VehicleAccelerationPublisher::publishVerticalChanged(float Vertical) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onVerticalChanged(Vertical);
    }
    std::shared_lock<std::shared_timed_mutex> verticalCallbacksLock(m_verticalCallbacksMutex);
    const auto verticalCallbacks = m_verticalCallbacks;
    verticalCallbacksLock.unlock();
    for(const auto& callbackEntry: verticalCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Vertical);
        }
    }
}

