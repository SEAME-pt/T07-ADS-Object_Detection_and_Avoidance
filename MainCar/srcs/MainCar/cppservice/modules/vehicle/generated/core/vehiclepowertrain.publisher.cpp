

#include "vehicle/generated/core/vehiclepowertrain.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehiclePowertrainPublisher::subscribeToAllChanges(IVehiclePowertrainSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehiclePowertrainSubscriber>(subscriber));
    }
}

void VehiclePowertrainPublisher::unsubscribeFromAllChanges(IVehiclePowertrainSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehiclePowertrainPublisher::subscribeToAccumulatedBrakingEnergyChanged(VehiclePowertrainAccumulatedBrakingEnergyPropertyCb callback)
{
    auto handleId = m_accumulatedBrakingEnergyChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_accumulatedBrakingEnergyCallbacksMutex);
    m_accumulatedBrakingEnergyCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainPublisher::unsubscribeFromAccumulatedBrakingEnergyChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_accumulatedBrakingEnergyCallbacksMutex);
    m_accumulatedBrakingEnergyCallbacks.erase(handleId);
}

void VehiclePowertrainPublisher::publishAccumulatedBrakingEnergyChanged(float AccumulatedBrakingEnergy) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onAccumulatedBrakingEnergyChanged(AccumulatedBrakingEnergy);
    }
    std::shared_lock<std::shared_timed_mutex> accumulatedBrakingEnergyCallbacksLock(m_accumulatedBrakingEnergyCallbacksMutex);
    const auto accumulatedBrakingEnergyCallbacks = m_accumulatedBrakingEnergyCallbacks;
    accumulatedBrakingEnergyCallbacksLock.unlock();
    for(const auto& callbackEntry: accumulatedBrakingEnergyCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(AccumulatedBrakingEnergy);
        }
    }
}

long VehiclePowertrainPublisher::subscribeToRangeChanged(VehiclePowertrainRangePropertyCb callback)
{
    auto handleId = m_rangeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_rangeCallbacksMutex);
    m_rangeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainPublisher::unsubscribeFromRangeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rangeCallbacksMutex);
    m_rangeCallbacks.erase(handleId);
}

void VehiclePowertrainPublisher::publishRangeChanged(int32_t Range) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onRangeChanged(Range);
    }
    std::shared_lock<std::shared_timed_mutex> rangeCallbacksLock(m_rangeCallbacksMutex);
    const auto rangeCallbacks = m_rangeCallbacks;
    rangeCallbacksLock.unlock();
    for(const auto& callbackEntry: rangeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Range);
        }
    }
}

long VehiclePowertrainPublisher::subscribeToTimeRemainingChanged(VehiclePowertrainTimeRemainingPropertyCb callback)
{
    auto handleId = m_timeRemainingChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_timeRemainingCallbacksMutex);
    m_timeRemainingCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainPublisher::unsubscribeFromTimeRemainingChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_timeRemainingCallbacksMutex);
    m_timeRemainingCallbacks.erase(handleId);
}

void VehiclePowertrainPublisher::publishTimeRemainingChanged(int32_t TimeRemaining) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTimeRemainingChanged(TimeRemaining);
    }
    std::shared_lock<std::shared_timed_mutex> timeRemainingCallbacksLock(m_timeRemainingCallbacksMutex);
    const auto timeRemainingCallbacks = m_timeRemainingCallbacks;
    timeRemainingCallbacksLock.unlock();
    for(const auto& callbackEntry: timeRemainingCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TimeRemaining);
        }
    }
}

long VehiclePowertrainPublisher::subscribeToTypeChanged(VehiclePowertrainTypePropertyCb callback)
{
    auto handleId = m_typeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_typeCallbacksMutex);
    m_typeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainPublisher::unsubscribeFromTypeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_typeCallbacksMutex);
    m_typeCallbacks.erase(handleId);
}

void VehiclePowertrainPublisher::publishTypeChanged(Vehicle_Powertrain_Type_ValueEnum Type) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTypeChanged(Type);
    }
    std::shared_lock<std::shared_timed_mutex> typeCallbacksLock(m_typeCallbacksMutex);
    const auto typeCallbacks = m_typeCallbacks;
    typeCallbacksLock.unlock();
    for(const auto& callbackEntry: typeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Type);
        }
    }
}

long VehiclePowertrainPublisher::subscribeToPowerOptimizeLevelChanged(VehiclePowertrainPowerOptimizeLevelPropertyCb callback)
{
    auto handleId = m_powerOptimizeLevelChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_powerOptimizeLevelCallbacksMutex);
    m_powerOptimizeLevelCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainPublisher::unsubscribeFromPowerOptimizeLevelChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_powerOptimizeLevelCallbacksMutex);
    m_powerOptimizeLevelCallbacks.erase(handleId);
}

void VehiclePowertrainPublisher::publishPowerOptimizeLevelChanged(int32_t PowerOptimizeLevel) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onPowerOptimizeLevelChanged(PowerOptimizeLevel);
    }
    std::shared_lock<std::shared_timed_mutex> powerOptimizeLevelCallbacksLock(m_powerOptimizeLevelCallbacksMutex);
    const auto powerOptimizeLevelCallbacks = m_powerOptimizeLevelCallbacks;
    powerOptimizeLevelCallbacksLock.unlock();
    for(const auto& callbackEntry: powerOptimizeLevelCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(PowerOptimizeLevel);
        }
    }
}

long VehiclePowertrainPublisher::subscribeToIsAutoPowerOptimizeChanged(VehiclePowertrainIsAutoPowerOptimizePropertyCb callback)
{
    auto handleId = m_isAutoPowerOptimizeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isAutoPowerOptimizeCallbacksMutex);
    m_isAutoPowerOptimizeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainPublisher::unsubscribeFromIsAutoPowerOptimizeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isAutoPowerOptimizeCallbacksMutex);
    m_isAutoPowerOptimizeCallbacks.erase(handleId);
}

void VehiclePowertrainPublisher::publishIsAutoPowerOptimizeChanged(bool IsAutoPowerOptimize) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsAutoPowerOptimizeChanged(IsAutoPowerOptimize);
    }
    std::shared_lock<std::shared_timed_mutex> isAutoPowerOptimizeCallbacksLock(m_isAutoPowerOptimizeCallbacksMutex);
    const auto isAutoPowerOptimizeCallbacks = m_isAutoPowerOptimizeCallbacks;
    isAutoPowerOptimizeCallbacksLock.unlock();
    for(const auto& callbackEntry: isAutoPowerOptimizeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsAutoPowerOptimize);
        }
    }
}

