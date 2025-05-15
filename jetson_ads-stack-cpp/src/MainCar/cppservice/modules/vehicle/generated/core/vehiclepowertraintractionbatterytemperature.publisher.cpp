

#include "vehicle/generated/core/vehiclepowertraintractionbatterytemperature.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehiclePowertrainTractionBatteryTemperaturePublisher::subscribeToAllChanges(IVehiclePowertrainTractionBatteryTemperatureSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehiclePowertrainTractionBatteryTemperatureSubscriber>(subscriber));
    }
}

void VehiclePowertrainTractionBatteryTemperaturePublisher::unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryTemperatureSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehiclePowertrainTractionBatteryTemperaturePublisher::subscribeToAverageChanged(VehiclePowertrainTractionBatteryTemperatureAveragePropertyCb callback)
{
    auto handleId = m_averageChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_averageCallbacksMutex);
    m_averageCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryTemperaturePublisher::unsubscribeFromAverageChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_averageCallbacksMutex);
    m_averageCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryTemperaturePublisher::publishAverageChanged(float Average) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onAverageChanged(Average);
    }
    std::shared_lock<std::shared_timed_mutex> averageCallbacksLock(m_averageCallbacksMutex);
    const auto averageCallbacks = m_averageCallbacks;
    averageCallbacksLock.unlock();
    for(const auto& callbackEntry: averageCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Average);
        }
    }
}

long VehiclePowertrainTractionBatteryTemperaturePublisher::subscribeToMinChanged(VehiclePowertrainTractionBatteryTemperatureMinPropertyCb callback)
{
    auto handleId = m_minChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_minCallbacksMutex);
    m_minCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryTemperaturePublisher::unsubscribeFromMinChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_minCallbacksMutex);
    m_minCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryTemperaturePublisher::publishMinChanged(float Min) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onMinChanged(Min);
    }
    std::shared_lock<std::shared_timed_mutex> minCallbacksLock(m_minCallbacksMutex);
    const auto minCallbacks = m_minCallbacks;
    minCallbacksLock.unlock();
    for(const auto& callbackEntry: minCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Min);
        }
    }
}

long VehiclePowertrainTractionBatteryTemperaturePublisher::subscribeToMaxChanged(VehiclePowertrainTractionBatteryTemperatureMaxPropertyCb callback)
{
    auto handleId = m_maxChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_maxCallbacksMutex);
    m_maxCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryTemperaturePublisher::unsubscribeFromMaxChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_maxCallbacksMutex);
    m_maxCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryTemperaturePublisher::publishMaxChanged(float Max) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onMaxChanged(Max);
    }
    std::shared_lock<std::shared_timed_mutex> maxCallbacksLock(m_maxCallbacksMutex);
    const auto maxCallbacks = m_maxCallbacks;
    maxCallbacksLock.unlock();
    for(const auto& callbackEntry: maxCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Max);
        }
    }
}

long VehiclePowertrainTractionBatteryTemperaturePublisher::subscribeToCellTemperatureChanged(VehiclePowertrainTractionBatteryTemperatureCellTemperaturePropertyCb callback)
{
    auto handleId = m_cellTemperatureChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_cellTemperatureCallbacksMutex);
    m_cellTemperatureCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryTemperaturePublisher::unsubscribeFromCellTemperatureChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_cellTemperatureCallbacksMutex);
    m_cellTemperatureCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryTemperaturePublisher::publishCellTemperatureChanged(const std::list<float>& CellTemperature) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onCellTemperatureChanged(CellTemperature);
    }
    std::shared_lock<std::shared_timed_mutex> cellTemperatureCallbacksLock(m_cellTemperatureCallbacksMutex);
    const auto cellTemperatureCallbacks = m_cellTemperatureCallbacks;
    cellTemperatureCallbacksLock.unlock();
    for(const auto& callbackEntry: cellTemperatureCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(CellTemperature);
        }
    }
}

