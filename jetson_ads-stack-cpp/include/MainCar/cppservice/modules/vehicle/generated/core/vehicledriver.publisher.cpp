

#include "vehicle/generated/core/vehicledriver.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleDriverPublisher::subscribeToAllChanges(IVehicleDriverSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleDriverSubscriber>(subscriber));
    }
}

void VehicleDriverPublisher::unsubscribeFromAllChanges(IVehicleDriverSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleDriverPublisher::subscribeToDistractionLevelChanged(VehicleDriverDistractionLevelPropertyCb callback)
{
    auto handleId = m_distractionLevelChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_distractionLevelCallbacksMutex);
    m_distractionLevelCallbacks[handleId] = callback;
    return handleId;
}

void VehicleDriverPublisher::unsubscribeFromDistractionLevelChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_distractionLevelCallbacksMutex);
    m_distractionLevelCallbacks.erase(handleId);
}

void VehicleDriverPublisher::publishDistractionLevelChanged(float DistractionLevel) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onDistractionLevelChanged(DistractionLevel);
    }
    std::shared_lock<std::shared_timed_mutex> distractionLevelCallbacksLock(m_distractionLevelCallbacksMutex);
    const auto distractionLevelCallbacks = m_distractionLevelCallbacks;
    distractionLevelCallbacksLock.unlock();
    for(const auto& callbackEntry: distractionLevelCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(DistractionLevel);
        }
    }
}

long VehicleDriverPublisher::subscribeToIsEyesOnRoadChanged(VehicleDriverIsEyesOnRoadPropertyCb callback)
{
    auto handleId = m_isEyesOnRoadChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isEyesOnRoadCallbacksMutex);
    m_isEyesOnRoadCallbacks[handleId] = callback;
    return handleId;
}

void VehicleDriverPublisher::unsubscribeFromIsEyesOnRoadChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isEyesOnRoadCallbacksMutex);
    m_isEyesOnRoadCallbacks.erase(handleId);
}

void VehicleDriverPublisher::publishIsEyesOnRoadChanged(bool IsEyesOnRoad) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsEyesOnRoadChanged(IsEyesOnRoad);
    }
    std::shared_lock<std::shared_timed_mutex> isEyesOnRoadCallbacksLock(m_isEyesOnRoadCallbacksMutex);
    const auto isEyesOnRoadCallbacks = m_isEyesOnRoadCallbacks;
    isEyesOnRoadCallbacksLock.unlock();
    for(const auto& callbackEntry: isEyesOnRoadCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsEyesOnRoad);
        }
    }
}

long VehicleDriverPublisher::subscribeToIsHandsOnWheelChanged(VehicleDriverIsHandsOnWheelPropertyCb callback)
{
    auto handleId = m_isHandsOnWheelChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isHandsOnWheelCallbacksMutex);
    m_isHandsOnWheelCallbacks[handleId] = callback;
    return handleId;
}

void VehicleDriverPublisher::unsubscribeFromIsHandsOnWheelChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isHandsOnWheelCallbacksMutex);
    m_isHandsOnWheelCallbacks.erase(handleId);
}

void VehicleDriverPublisher::publishIsHandsOnWheelChanged(bool IsHandsOnWheel) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsHandsOnWheelChanged(IsHandsOnWheel);
    }
    std::shared_lock<std::shared_timed_mutex> isHandsOnWheelCallbacksLock(m_isHandsOnWheelCallbacksMutex);
    const auto isHandsOnWheelCallbacks = m_isHandsOnWheelCallbacks;
    isHandsOnWheelCallbacksLock.unlock();
    for(const auto& callbackEntry: isHandsOnWheelCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsHandsOnWheel);
        }
    }
}

long VehicleDriverPublisher::subscribeToAttentiveProbabilityChanged(VehicleDriverAttentiveProbabilityPropertyCb callback)
{
    auto handleId = m_attentiveProbabilityChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_attentiveProbabilityCallbacksMutex);
    m_attentiveProbabilityCallbacks[handleId] = callback;
    return handleId;
}

void VehicleDriverPublisher::unsubscribeFromAttentiveProbabilityChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_attentiveProbabilityCallbacksMutex);
    m_attentiveProbabilityCallbacks.erase(handleId);
}

void VehicleDriverPublisher::publishAttentiveProbabilityChanged(float AttentiveProbability) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onAttentiveProbabilityChanged(AttentiveProbability);
    }
    std::shared_lock<std::shared_timed_mutex> attentiveProbabilityCallbacksLock(m_attentiveProbabilityCallbacksMutex);
    const auto attentiveProbabilityCallbacks = m_attentiveProbabilityCallbacks;
    attentiveProbabilityCallbacksLock.unlock();
    for(const auto& callbackEntry: attentiveProbabilityCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(AttentiveProbability);
        }
    }
}

long VehicleDriverPublisher::subscribeToFatigueLevelChanged(VehicleDriverFatigueLevelPropertyCb callback)
{
    auto handleId = m_fatigueLevelChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_fatigueLevelCallbacksMutex);
    m_fatigueLevelCallbacks[handleId] = callback;
    return handleId;
}

void VehicleDriverPublisher::unsubscribeFromFatigueLevelChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_fatigueLevelCallbacksMutex);
    m_fatigueLevelCallbacks.erase(handleId);
}

void VehicleDriverPublisher::publishFatigueLevelChanged(float FatigueLevel) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onFatigueLevelChanged(FatigueLevel);
    }
    std::shared_lock<std::shared_timed_mutex> fatigueLevelCallbacksLock(m_fatigueLevelCallbacksMutex);
    const auto fatigueLevelCallbacks = m_fatigueLevelCallbacks;
    fatigueLevelCallbacksLock.unlock();
    for(const auto& callbackEntry: fatigueLevelCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(FatigueLevel);
        }
    }
}

long VehicleDriverPublisher::subscribeToHeartRateChanged(VehicleDriverHeartRatePropertyCb callback)
{
    auto handleId = m_heartRateChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_heartRateCallbacksMutex);
    m_heartRateCallbacks[handleId] = callback;
    return handleId;
}

void VehicleDriverPublisher::unsubscribeFromHeartRateChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_heartRateCallbacksMutex);
    m_heartRateCallbacks.erase(handleId);
}

void VehicleDriverPublisher::publishHeartRateChanged(int32_t HeartRate) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onHeartRateChanged(HeartRate);
    }
    std::shared_lock<std::shared_timed_mutex> heartRateCallbacksLock(m_heartRateCallbacksMutex);
    const auto heartRateCallbacks = m_heartRateCallbacks;
    heartRateCallbacksLock.unlock();
    for(const auto& callbackEntry: heartRateCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(HeartRate);
        }
    }
}

