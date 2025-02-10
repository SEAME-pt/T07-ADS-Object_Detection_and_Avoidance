

#include "vehicle/generated/core/vehiclebodylights.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleBodyLightsPublisher::subscribeToAllChanges(IVehicleBodyLightsSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleBodyLightsSubscriber>(subscriber));
    }
}

void VehicleBodyLightsPublisher::unsubscribeFromAllChanges(IVehicleBodyLightsSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleBodyLightsPublisher::subscribeToLightSwitchChanged(VehicleBodyLightsLightSwitchPropertyCb callback)
{
    auto handleId = m_lightSwitchChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_lightSwitchCallbacksMutex);
    m_lightSwitchCallbacks[handleId] = callback;
    return handleId;
}

void VehicleBodyLightsPublisher::unsubscribeFromLightSwitchChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_lightSwitchCallbacksMutex);
    m_lightSwitchCallbacks.erase(handleId);
}

void VehicleBodyLightsPublisher::publishLightSwitchChanged(Vehicle_Body_Lights_LightSwitch_ValueEnum LightSwitch) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onLightSwitchChanged(LightSwitch);
    }
    std::shared_lock<std::shared_timed_mutex> lightSwitchCallbacksLock(m_lightSwitchCallbacksMutex);
    const auto lightSwitchCallbacks = m_lightSwitchCallbacks;
    lightSwitchCallbacksLock.unlock();
    for(const auto& callbackEntry: lightSwitchCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(LightSwitch);
        }
    }
}

long VehicleBodyLightsPublisher::subscribeToIsHighBeamSwitchOnChanged(VehicleBodyLightsIsHighBeamSwitchOnPropertyCb callback)
{
    auto handleId = m_isHighBeamSwitchOnChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isHighBeamSwitchOnCallbacksMutex);
    m_isHighBeamSwitchOnCallbacks[handleId] = callback;
    return handleId;
}

void VehicleBodyLightsPublisher::unsubscribeFromIsHighBeamSwitchOnChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isHighBeamSwitchOnCallbacksMutex);
    m_isHighBeamSwitchOnCallbacks.erase(handleId);
}

void VehicleBodyLightsPublisher::publishIsHighBeamSwitchOnChanged(bool IsHighBeamSwitchOn) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsHighBeamSwitchOnChanged(IsHighBeamSwitchOn);
    }
    std::shared_lock<std::shared_timed_mutex> isHighBeamSwitchOnCallbacksLock(m_isHighBeamSwitchOnCallbacksMutex);
    const auto isHighBeamSwitchOnCallbacks = m_isHighBeamSwitchOnCallbacks;
    isHighBeamSwitchOnCallbacksLock.unlock();
    for(const auto& callbackEntry: isHighBeamSwitchOnCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsHighBeamSwitchOn);
        }
    }
}

