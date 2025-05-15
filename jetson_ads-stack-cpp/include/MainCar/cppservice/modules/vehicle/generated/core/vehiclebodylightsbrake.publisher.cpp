

#include "vehicle/generated/core/vehiclebodylightsbrake.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleBodyLightsBrakePublisher::subscribeToAllChanges(IVehicleBodyLightsBrakeSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleBodyLightsBrakeSubscriber>(subscriber));
    }
}

void VehicleBodyLightsBrakePublisher::unsubscribeFromAllChanges(IVehicleBodyLightsBrakeSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleBodyLightsBrakePublisher::subscribeToIsActiveChanged(VehicleBodyLightsBrakeIsActivePropertyCb callback)
{
    auto handleId = m_isActiveChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isActiveCallbacksMutex);
    m_isActiveCallbacks[handleId] = callback;
    return handleId;
}

void VehicleBodyLightsBrakePublisher::unsubscribeFromIsActiveChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isActiveCallbacksMutex);
    m_isActiveCallbacks.erase(handleId);
}

void VehicleBodyLightsBrakePublisher::publishIsActiveChanged(Vehicle_Body_Lights_Brake_IsActive_ValueEnum IsActive) const
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

long VehicleBodyLightsBrakePublisher::subscribeToIsDefectChanged(VehicleBodyLightsBrakeIsDefectPropertyCb callback)
{
    auto handleId = m_isDefectChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectCallbacksMutex);
    m_isDefectCallbacks[handleId] = callback;
    return handleId;
}

void VehicleBodyLightsBrakePublisher::unsubscribeFromIsDefectChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isDefectCallbacksMutex);
    m_isDefectCallbacks.erase(handleId);
}

void VehicleBodyLightsBrakePublisher::publishIsDefectChanged(bool IsDefect) const
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

