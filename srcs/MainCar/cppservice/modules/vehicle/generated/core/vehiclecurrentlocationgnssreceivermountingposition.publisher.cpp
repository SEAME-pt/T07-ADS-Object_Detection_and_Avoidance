

#include "vehicle/generated/core/vehiclecurrentlocationgnssreceivermountingposition.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleCurrentLocationGnssReceiverMountingPositionPublisher::subscribeToAllChanges(IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber>(subscriber));
    }
}

void VehicleCurrentLocationGnssReceiverMountingPositionPublisher::unsubscribeFromAllChanges(IVehicleCurrentLocationGnssReceiverMountingPositionSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleCurrentLocationGnssReceiverMountingPositionPublisher::subscribeToXChanged(VehicleCurrentLocationGnssReceiverMountingPositionXPropertyCb callback)
{
    auto handleId = m_xChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_xCallbacksMutex);
    m_xCallbacks[handleId] = callback;
    return handleId;
}

void VehicleCurrentLocationGnssReceiverMountingPositionPublisher::unsubscribeFromXChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_xCallbacksMutex);
    m_xCallbacks.erase(handleId);
}

void VehicleCurrentLocationGnssReceiverMountingPositionPublisher::publishXChanged(int32_t X) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onXChanged(X);
    }
    std::shared_lock<std::shared_timed_mutex> xCallbacksLock(m_xCallbacksMutex);
    const auto xCallbacks = m_xCallbacks;
    xCallbacksLock.unlock();
    for(const auto& callbackEntry: xCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(X);
        }
    }
}

long VehicleCurrentLocationGnssReceiverMountingPositionPublisher::subscribeToYChanged(VehicleCurrentLocationGnssReceiverMountingPositionYPropertyCb callback)
{
    auto handleId = m_yChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_yCallbacksMutex);
    m_yCallbacks[handleId] = callback;
    return handleId;
}

void VehicleCurrentLocationGnssReceiverMountingPositionPublisher::unsubscribeFromYChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_yCallbacksMutex);
    m_yCallbacks.erase(handleId);
}

void VehicleCurrentLocationGnssReceiverMountingPositionPublisher::publishYChanged(int32_t Y) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onYChanged(Y);
    }
    std::shared_lock<std::shared_timed_mutex> yCallbacksLock(m_yCallbacksMutex);
    const auto yCallbacks = m_yCallbacks;
    yCallbacksLock.unlock();
    for(const auto& callbackEntry: yCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Y);
        }
    }
}

long VehicleCurrentLocationGnssReceiverMountingPositionPublisher::subscribeToZChanged(VehicleCurrentLocationGnssReceiverMountingPositionZPropertyCb callback)
{
    auto handleId = m_zChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_zCallbacksMutex);
    m_zCallbacks[handleId] = callback;
    return handleId;
}

void VehicleCurrentLocationGnssReceiverMountingPositionPublisher::unsubscribeFromZChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_zCallbacksMutex);
    m_zCallbacks.erase(handleId);
}

void VehicleCurrentLocationGnssReceiverMountingPositionPublisher::publishZChanged(int32_t Z) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onZChanged(Z);
    }
    std::shared_lock<std::shared_timed_mutex> zCallbacksLock(m_zCallbacksMutex);
    const auto zCallbacks = m_zCallbacks;
    zCallbacksLock.unlock();
    for(const auto& callbackEntry: zCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Z);
        }
    }
}

