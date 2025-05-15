

#include "vehicle/generated/core/vehiclecurrentlocation.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleCurrentLocationPublisher::subscribeToAllChanges(IVehicleCurrentLocationSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleCurrentLocationSubscriber>(subscriber));
    }
}

void VehicleCurrentLocationPublisher::unsubscribeFromAllChanges(IVehicleCurrentLocationSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleCurrentLocationPublisher::subscribeToTimestampChanged(VehicleCurrentLocationTimestampPropertyCb callback)
{
    auto handleId = m_timestampChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_timestampCallbacksMutex);
    m_timestampCallbacks[handleId] = callback;
    return handleId;
}

void VehicleCurrentLocationPublisher::unsubscribeFromTimestampChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_timestampCallbacksMutex);
    m_timestampCallbacks.erase(handleId);
}

void VehicleCurrentLocationPublisher::publishTimestampChanged(const std::string& Timestamp) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTimestampChanged(Timestamp);
    }
    std::shared_lock<std::shared_timed_mutex> timestampCallbacksLock(m_timestampCallbacksMutex);
    const auto timestampCallbacks = m_timestampCallbacks;
    timestampCallbacksLock.unlock();
    for(const auto& callbackEntry: timestampCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Timestamp);
        }
    }
}

long VehicleCurrentLocationPublisher::subscribeToLatitudeChanged(VehicleCurrentLocationLatitudePropertyCb callback)
{
    auto handleId = m_latitudeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_latitudeCallbacksMutex);
    m_latitudeCallbacks[handleId] = callback;
    return handleId;
}

void VehicleCurrentLocationPublisher::unsubscribeFromLatitudeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_latitudeCallbacksMutex);
    m_latitudeCallbacks.erase(handleId);
}

void VehicleCurrentLocationPublisher::publishLatitudeChanged(double Latitude) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onLatitudeChanged(Latitude);
    }
    std::shared_lock<std::shared_timed_mutex> latitudeCallbacksLock(m_latitudeCallbacksMutex);
    const auto latitudeCallbacks = m_latitudeCallbacks;
    latitudeCallbacksLock.unlock();
    for(const auto& callbackEntry: latitudeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Latitude);
        }
    }
}

long VehicleCurrentLocationPublisher::subscribeToLongitudeChanged(VehicleCurrentLocationLongitudePropertyCb callback)
{
    auto handleId = m_longitudeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_longitudeCallbacksMutex);
    m_longitudeCallbacks[handleId] = callback;
    return handleId;
}

void VehicleCurrentLocationPublisher::unsubscribeFromLongitudeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_longitudeCallbacksMutex);
    m_longitudeCallbacks.erase(handleId);
}

void VehicleCurrentLocationPublisher::publishLongitudeChanged(double Longitude) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onLongitudeChanged(Longitude);
    }
    std::shared_lock<std::shared_timed_mutex> longitudeCallbacksLock(m_longitudeCallbacksMutex);
    const auto longitudeCallbacks = m_longitudeCallbacks;
    longitudeCallbacksLock.unlock();
    for(const auto& callbackEntry: longitudeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Longitude);
        }
    }
}

long VehicleCurrentLocationPublisher::subscribeToHeadingChanged(VehicleCurrentLocationHeadingPropertyCb callback)
{
    auto handleId = m_headingChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_headingCallbacksMutex);
    m_headingCallbacks[handleId] = callback;
    return handleId;
}

void VehicleCurrentLocationPublisher::unsubscribeFromHeadingChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_headingCallbacksMutex);
    m_headingCallbacks.erase(handleId);
}

void VehicleCurrentLocationPublisher::publishHeadingChanged(double Heading) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onHeadingChanged(Heading);
    }
    std::shared_lock<std::shared_timed_mutex> headingCallbacksLock(m_headingCallbacksMutex);
    const auto headingCallbacks = m_headingCallbacks;
    headingCallbacksLock.unlock();
    for(const auto& callbackEntry: headingCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Heading);
        }
    }
}

long VehicleCurrentLocationPublisher::subscribeToHorizontalAccuracyChanged(VehicleCurrentLocationHorizontalAccuracyPropertyCb callback)
{
    auto handleId = m_horizontalAccuracyChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_horizontalAccuracyCallbacksMutex);
    m_horizontalAccuracyCallbacks[handleId] = callback;
    return handleId;
}

void VehicleCurrentLocationPublisher::unsubscribeFromHorizontalAccuracyChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_horizontalAccuracyCallbacksMutex);
    m_horizontalAccuracyCallbacks.erase(handleId);
}

void VehicleCurrentLocationPublisher::publishHorizontalAccuracyChanged(double HorizontalAccuracy) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onHorizontalAccuracyChanged(HorizontalAccuracy);
    }
    std::shared_lock<std::shared_timed_mutex> horizontalAccuracyCallbacksLock(m_horizontalAccuracyCallbacksMutex);
    const auto horizontalAccuracyCallbacks = m_horizontalAccuracyCallbacks;
    horizontalAccuracyCallbacksLock.unlock();
    for(const auto& callbackEntry: horizontalAccuracyCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(HorizontalAccuracy);
        }
    }
}

long VehicleCurrentLocationPublisher::subscribeToAltitudeChanged(VehicleCurrentLocationAltitudePropertyCb callback)
{
    auto handleId = m_altitudeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_altitudeCallbacksMutex);
    m_altitudeCallbacks[handleId] = callback;
    return handleId;
}

void VehicleCurrentLocationPublisher::unsubscribeFromAltitudeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_altitudeCallbacksMutex);
    m_altitudeCallbacks.erase(handleId);
}

void VehicleCurrentLocationPublisher::publishAltitudeChanged(double Altitude) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onAltitudeChanged(Altitude);
    }
    std::shared_lock<std::shared_timed_mutex> altitudeCallbacksLock(m_altitudeCallbacksMutex);
    const auto altitudeCallbacks = m_altitudeCallbacks;
    altitudeCallbacksLock.unlock();
    for(const auto& callbackEntry: altitudeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Altitude);
        }
    }
}

long VehicleCurrentLocationPublisher::subscribeToVerticalAccuracyChanged(VehicleCurrentLocationVerticalAccuracyPropertyCb callback)
{
    auto handleId = m_verticalAccuracyChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_verticalAccuracyCallbacksMutex);
    m_verticalAccuracyCallbacks[handleId] = callback;
    return handleId;
}

void VehicleCurrentLocationPublisher::unsubscribeFromVerticalAccuracyChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_verticalAccuracyCallbacksMutex);
    m_verticalAccuracyCallbacks.erase(handleId);
}

void VehicleCurrentLocationPublisher::publishVerticalAccuracyChanged(double VerticalAccuracy) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onVerticalAccuracyChanged(VerticalAccuracy);
    }
    std::shared_lock<std::shared_timed_mutex> verticalAccuracyCallbacksLock(m_verticalAccuracyCallbacksMutex);
    const auto verticalAccuracyCallbacks = m_verticalAccuracyCallbacks;
    verticalAccuracyCallbacksLock.unlock();
    for(const auto& callbackEntry: verticalAccuracyCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(VerticalAccuracy);
        }
    }
}

