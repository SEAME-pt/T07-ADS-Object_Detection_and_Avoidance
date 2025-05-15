

#include "vehicle/generated/core/vehiclepowertraintractionbatterychargingmaximumchargingcurrent.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber>(subscriber));
    }
}

void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingMaximumChargingCurrentSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::subscribeToDcChanged(VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentDcPropertyCb callback)
{
    auto handleId = m_dcChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_dcCallbacksMutex);
    m_dcCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::unsubscribeFromDcChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_dcCallbacksMutex);
    m_dcCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::publishDcChanged(float DC) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onDcChanged(DC);
    }
    std::shared_lock<std::shared_timed_mutex> dcCallbacksLock(m_dcCallbacksMutex);
    const auto dcCallbacks = m_dcCallbacks;
    dcCallbacksLock.unlock();
    for(const auto& callbackEntry: dcCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(DC);
        }
    }
}

long VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::subscribeToPhase1Changed(VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase1PropertyCb callback)
{
    auto handleId = m_phase1ChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_phase1CallbacksMutex);
    m_phase1Callbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::unsubscribeFromPhase1Changed(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_phase1CallbacksMutex);
    m_phase1Callbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::publishPhase1Changed(float Phase1) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onPhase1Changed(Phase1);
    }
    std::shared_lock<std::shared_timed_mutex> phase1CallbacksLock(m_phase1CallbacksMutex);
    const auto phase1Callbacks = m_phase1Callbacks;
    phase1CallbacksLock.unlock();
    for(const auto& callbackEntry: phase1Callbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Phase1);
        }
    }
}

long VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::subscribeToPhase2Changed(VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase2PropertyCb callback)
{
    auto handleId = m_phase2ChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_phase2CallbacksMutex);
    m_phase2Callbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::unsubscribeFromPhase2Changed(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_phase2CallbacksMutex);
    m_phase2Callbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::publishPhase2Changed(float Phase2) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onPhase2Changed(Phase2);
    }
    std::shared_lock<std::shared_timed_mutex> phase2CallbacksLock(m_phase2CallbacksMutex);
    const auto phase2Callbacks = m_phase2Callbacks;
    phase2CallbacksLock.unlock();
    for(const auto& callbackEntry: phase2Callbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Phase2);
        }
    }
}

long VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::subscribeToPhase3Changed(VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPhase3PropertyCb callback)
{
    auto handleId = m_phase3ChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_phase3CallbacksMutex);
    m_phase3Callbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::unsubscribeFromPhase3Changed(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_phase3CallbacksMutex);
    m_phase3Callbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryChargingMaximumChargingCurrentPublisher::publishPhase3Changed(float Phase3) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onPhase3Changed(Phase3);
    }
    std::shared_lock<std::shared_timed_mutex> phase3CallbacksLock(m_phase3CallbacksMutex);
    const auto phase3Callbacks = m_phase3Callbacks;
    phase3CallbacksLock.unlock();
    for(const auto& callbackEntry: phase3Callbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Phase3);
        }
    }
}

