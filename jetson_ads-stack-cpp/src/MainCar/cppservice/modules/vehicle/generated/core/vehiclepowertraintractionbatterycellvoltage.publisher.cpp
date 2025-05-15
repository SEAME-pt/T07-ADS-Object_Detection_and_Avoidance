

#include "vehicle/generated/core/vehiclepowertraintractionbatterycellvoltage.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehiclePowertrainTractionBatteryCellVoltagePublisher::subscribeToAllChanges(IVehiclePowertrainTractionBatteryCellVoltageSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehiclePowertrainTractionBatteryCellVoltageSubscriber>(subscriber));
    }
}

void VehiclePowertrainTractionBatteryCellVoltagePublisher::unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryCellVoltageSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehiclePowertrainTractionBatteryCellVoltagePublisher::subscribeToMinChanged(VehiclePowertrainTractionBatteryCellVoltageMinPropertyCb callback)
{
    auto handleId = m_minChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_minCallbacksMutex);
    m_minCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryCellVoltagePublisher::unsubscribeFromMinChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_minCallbacksMutex);
    m_minCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryCellVoltagePublisher::publishMinChanged(float Min) const
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

long VehiclePowertrainTractionBatteryCellVoltagePublisher::subscribeToIdMinChanged(VehiclePowertrainTractionBatteryCellVoltageIdMinPropertyCb callback)
{
    auto handleId = m_idMinChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_idMinCallbacksMutex);
    m_idMinCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryCellVoltagePublisher::unsubscribeFromIdMinChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_idMinCallbacksMutex);
    m_idMinCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryCellVoltagePublisher::publishIdMinChanged(int32_t IdMin) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIdMinChanged(IdMin);
    }
    std::shared_lock<std::shared_timed_mutex> idMinCallbacksLock(m_idMinCallbacksMutex);
    const auto idMinCallbacks = m_idMinCallbacks;
    idMinCallbacksLock.unlock();
    for(const auto& callbackEntry: idMinCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IdMin);
        }
    }
}

long VehiclePowertrainTractionBatteryCellVoltagePublisher::subscribeToMaxChanged(VehiclePowertrainTractionBatteryCellVoltageMaxPropertyCb callback)
{
    auto handleId = m_maxChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_maxCallbacksMutex);
    m_maxCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryCellVoltagePublisher::unsubscribeFromMaxChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_maxCallbacksMutex);
    m_maxCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryCellVoltagePublisher::publishMaxChanged(float Max) const
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

long VehiclePowertrainTractionBatteryCellVoltagePublisher::subscribeToIdMaxChanged(VehiclePowertrainTractionBatteryCellVoltageIdMaxPropertyCb callback)
{
    auto handleId = m_idMaxChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_idMaxCallbacksMutex);
    m_idMaxCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryCellVoltagePublisher::unsubscribeFromIdMaxChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_idMaxCallbacksMutex);
    m_idMaxCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryCellVoltagePublisher::publishIdMaxChanged(int32_t IdMax) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIdMaxChanged(IdMax);
    }
    std::shared_lock<std::shared_timed_mutex> idMaxCallbacksLock(m_idMaxCallbacksMutex);
    const auto idMaxCallbacks = m_idMaxCallbacks;
    idMaxCallbacksLock.unlock();
    for(const auto& callbackEntry: idMaxCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IdMax);
        }
    }
}

long VehiclePowertrainTractionBatteryCellVoltagePublisher::subscribeToCellVoltagesChanged(VehiclePowertrainTractionBatteryCellVoltageCellVoltagesPropertyCb callback)
{
    auto handleId = m_cellVoltagesChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_cellVoltagesCallbacksMutex);
    m_cellVoltagesCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryCellVoltagePublisher::unsubscribeFromCellVoltagesChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_cellVoltagesCallbacksMutex);
    m_cellVoltagesCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryCellVoltagePublisher::publishCellVoltagesChanged(const std::list<float>& CellVoltages) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onCellVoltagesChanged(CellVoltages);
    }
    std::shared_lock<std::shared_timed_mutex> cellVoltagesCallbacksLock(m_cellVoltagesCallbacksMutex);
    const auto cellVoltagesCallbacks = m_cellVoltagesCallbacks;
    cellVoltagesCallbacksLock.unlock();
    for(const auto& callbackEntry: cellVoltagesCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(CellVoltages);
        }
    }
}

