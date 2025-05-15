

#include "vehicle/generated/core/vehicleangularvelocity.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleAngularVelocityPublisher::subscribeToAllChanges(IVehicleAngularVelocitySubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleAngularVelocitySubscriber>(subscriber));
    }
}

void VehicleAngularVelocityPublisher::unsubscribeFromAllChanges(IVehicleAngularVelocitySubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleAngularVelocityPublisher::subscribeToRollChanged(VehicleAngularVelocityRollPropertyCb callback)
{
    auto handleId = m_rollChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_rollCallbacksMutex);
    m_rollCallbacks[handleId] = callback;
    return handleId;
}

void VehicleAngularVelocityPublisher::unsubscribeFromRollChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rollCallbacksMutex);
    m_rollCallbacks.erase(handleId);
}

void VehicleAngularVelocityPublisher::publishRollChanged(float Roll) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onRollChanged(Roll);
    }
    std::shared_lock<std::shared_timed_mutex> rollCallbacksLock(m_rollCallbacksMutex);
    const auto rollCallbacks = m_rollCallbacks;
    rollCallbacksLock.unlock();
    for(const auto& callbackEntry: rollCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Roll);
        }
    }
}

long VehicleAngularVelocityPublisher::subscribeToPitchChanged(VehicleAngularVelocityPitchPropertyCb callback)
{
    auto handleId = m_pitchChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_pitchCallbacksMutex);
    m_pitchCallbacks[handleId] = callback;
    return handleId;
}

void VehicleAngularVelocityPublisher::unsubscribeFromPitchChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_pitchCallbacksMutex);
    m_pitchCallbacks.erase(handleId);
}

void VehicleAngularVelocityPublisher::publishPitchChanged(float Pitch) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onPitchChanged(Pitch);
    }
    std::shared_lock<std::shared_timed_mutex> pitchCallbacksLock(m_pitchCallbacksMutex);
    const auto pitchCallbacks = m_pitchCallbacks;
    pitchCallbacksLock.unlock();
    for(const auto& callbackEntry: pitchCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Pitch);
        }
    }
}

long VehicleAngularVelocityPublisher::subscribeToYawChanged(VehicleAngularVelocityYawPropertyCb callback)
{
    auto handleId = m_yawChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_yawCallbacksMutex);
    m_yawCallbacks[handleId] = callback;
    return handleId;
}

void VehicleAngularVelocityPublisher::unsubscribeFromYawChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_yawCallbacksMutex);
    m_yawCallbacks.erase(handleId);
}

void VehicleAngularVelocityPublisher::publishYawChanged(float Yaw) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onYawChanged(Yaw);
    }
    std::shared_lock<std::shared_timed_mutex> yawCallbacksLock(m_yawCallbacksMutex);
    const auto yawCallbacks = m_yawCallbacks;
    yawCallbacksLock.unlock();
    for(const auto& callbackEntry: yawCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Yaw);
        }
    }
}

