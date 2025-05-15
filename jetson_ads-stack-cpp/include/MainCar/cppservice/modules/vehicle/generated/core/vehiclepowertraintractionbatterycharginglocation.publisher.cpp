

#include "vehicle/generated/core/vehiclepowertraintractionbatterycharginglocation.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehiclePowertrainTractionBatteryChargingLocationPublisher::subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingLocationSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehiclePowertrainTractionBatteryChargingLocationSubscriber>(subscriber));
    }
}

void VehiclePowertrainTractionBatteryChargingLocationPublisher::unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingLocationSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehiclePowertrainTractionBatteryChargingLocationPublisher::subscribeToLatitudeChanged(VehiclePowertrainTractionBatteryChargingLocationLatitudePropertyCb callback)
{
    auto handleId = m_latitudeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_latitudeCallbacksMutex);
    m_latitudeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingLocationPublisher::unsubscribeFromLatitudeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_latitudeCallbacksMutex);
    m_latitudeCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingLocationPublisher::publishLatitudeChanged(double Latitude) const
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

long VehiclePowertrainTractionBatteryChargingLocationPublisher::subscribeToLongitudeChanged(VehiclePowertrainTractionBatteryChargingLocationLongitudePropertyCb callback)
{
    auto handleId = m_longitudeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_longitudeCallbacksMutex);
    m_longitudeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingLocationPublisher::unsubscribeFromLongitudeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_longitudeCallbacksMutex);
    m_longitudeCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingLocationPublisher::publishLongitudeChanged(double Longitude) const
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

long VehiclePowertrainTractionBatteryChargingLocationPublisher::subscribeToAltitudeChanged(VehiclePowertrainTractionBatteryChargingLocationAltitudePropertyCb callback)
{
    auto handleId = m_altitudeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_altitudeCallbacksMutex);
    m_altitudeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingLocationPublisher::unsubscribeFromAltitudeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_altitudeCallbacksMutex);
    m_altitudeCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingLocationPublisher::publishAltitudeChanged(double Altitude) const
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

