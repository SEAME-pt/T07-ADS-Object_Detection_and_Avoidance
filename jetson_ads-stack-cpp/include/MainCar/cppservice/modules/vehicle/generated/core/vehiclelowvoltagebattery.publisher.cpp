

#include "vehicle/generated/core/vehiclelowvoltagebattery.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleLowVoltageBatteryPublisher::subscribeToAllChanges(IVehicleLowVoltageBatterySubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleLowVoltageBatterySubscriber>(subscriber));
    }
}

void VehicleLowVoltageBatteryPublisher::unsubscribeFromAllChanges(IVehicleLowVoltageBatterySubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleLowVoltageBatteryPublisher::subscribeToNominalVoltageChanged(VehicleLowVoltageBatteryNominalVoltagePropertyCb callback)
{
    auto handleId = m_nominalVoltageChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_nominalVoltageCallbacksMutex);
    m_nominalVoltageCallbacks[handleId] = callback;
    return handleId;
}

void VehicleLowVoltageBatteryPublisher::unsubscribeFromNominalVoltageChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_nominalVoltageCallbacksMutex);
    m_nominalVoltageCallbacks.erase(handleId);
}

void VehicleLowVoltageBatteryPublisher::publishNominalVoltageChanged(int32_t NominalVoltage) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onNominalVoltageChanged(NominalVoltage);
    }
    std::shared_lock<std::shared_timed_mutex> nominalVoltageCallbacksLock(m_nominalVoltageCallbacksMutex);
    const auto nominalVoltageCallbacks = m_nominalVoltageCallbacks;
    nominalVoltageCallbacksLock.unlock();
    for(const auto& callbackEntry: nominalVoltageCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(NominalVoltage);
        }
    }
}

long VehicleLowVoltageBatteryPublisher::subscribeToNominalCapacityChanged(VehicleLowVoltageBatteryNominalCapacityPropertyCb callback)
{
    auto handleId = m_nominalCapacityChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_nominalCapacityCallbacksMutex);
    m_nominalCapacityCallbacks[handleId] = callback;
    return handleId;
}

void VehicleLowVoltageBatteryPublisher::unsubscribeFromNominalCapacityChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_nominalCapacityCallbacksMutex);
    m_nominalCapacityCallbacks.erase(handleId);
}

void VehicleLowVoltageBatteryPublisher::publishNominalCapacityChanged(int32_t NominalCapacity) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onNominalCapacityChanged(NominalCapacity);
    }
    std::shared_lock<std::shared_timed_mutex> nominalCapacityCallbacksLock(m_nominalCapacityCallbacksMutex);
    const auto nominalCapacityCallbacks = m_nominalCapacityCallbacks;
    nominalCapacityCallbacksLock.unlock();
    for(const auto& callbackEntry: nominalCapacityCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(NominalCapacity);
        }
    }
}

long VehicleLowVoltageBatteryPublisher::subscribeToCurrentVoltageChanged(VehicleLowVoltageBatteryCurrentVoltagePropertyCb callback)
{
    auto handleId = m_currentVoltageChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_currentVoltageCallbacksMutex);
    m_currentVoltageCallbacks[handleId] = callback;
    return handleId;
}

void VehicleLowVoltageBatteryPublisher::unsubscribeFromCurrentVoltageChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_currentVoltageCallbacksMutex);
    m_currentVoltageCallbacks.erase(handleId);
}

void VehicleLowVoltageBatteryPublisher::publishCurrentVoltageChanged(float CurrentVoltage) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onCurrentVoltageChanged(CurrentVoltage);
    }
    std::shared_lock<std::shared_timed_mutex> currentVoltageCallbacksLock(m_currentVoltageCallbacksMutex);
    const auto currentVoltageCallbacks = m_currentVoltageCallbacks;
    currentVoltageCallbacksLock.unlock();
    for(const auto& callbackEntry: currentVoltageCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(CurrentVoltage);
        }
    }
}

long VehicleLowVoltageBatteryPublisher::subscribeToCurrentCurrentChanged(VehicleLowVoltageBatteryCurrentCurrentPropertyCb callback)
{
    auto handleId = m_currentCurrentChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_currentCurrentCallbacksMutex);
    m_currentCurrentCallbacks[handleId] = callback;
    return handleId;
}

void VehicleLowVoltageBatteryPublisher::unsubscribeFromCurrentCurrentChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_currentCurrentCallbacksMutex);
    m_currentCurrentCallbacks.erase(handleId);
}

void VehicleLowVoltageBatteryPublisher::publishCurrentCurrentChanged(float CurrentCurrent) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onCurrentCurrentChanged(CurrentCurrent);
    }
    std::shared_lock<std::shared_timed_mutex> currentCurrentCallbacksLock(m_currentCurrentCallbacksMutex);
    const auto currentCurrentCallbacks = m_currentCurrentCallbacks;
    currentCurrentCallbacksLock.unlock();
    for(const auto& callbackEntry: currentCurrentCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(CurrentCurrent);
        }
    }
}

