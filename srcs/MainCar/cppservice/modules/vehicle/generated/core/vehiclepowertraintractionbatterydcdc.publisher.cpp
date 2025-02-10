

#include "vehicle/generated/core/vehiclepowertraintractionbatterydcdc.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehiclePowertrainTractionBatteryDcdcPublisher::subscribeToAllChanges(IVehiclePowertrainTractionBatteryDcdcSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehiclePowertrainTractionBatteryDcdcSubscriber>(subscriber));
    }
}

void VehiclePowertrainTractionBatteryDcdcPublisher::unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryDcdcSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehiclePowertrainTractionBatteryDcdcPublisher::subscribeToPowerLossChanged(VehiclePowertrainTractionBatteryDcdcPowerLossPropertyCb callback)
{
    auto handleId = m_powerLossChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_powerLossCallbacksMutex);
    m_powerLossCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryDcdcPublisher::unsubscribeFromPowerLossChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_powerLossCallbacksMutex);
    m_powerLossCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryDcdcPublisher::publishPowerLossChanged(float PowerLoss) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onPowerLossChanged(PowerLoss);
    }
    std::shared_lock<std::shared_timed_mutex> powerLossCallbacksLock(m_powerLossCallbacksMutex);
    const auto powerLossCallbacks = m_powerLossCallbacks;
    powerLossCallbacksLock.unlock();
    for(const auto& callbackEntry: powerLossCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(PowerLoss);
        }
    }
}

long VehiclePowertrainTractionBatteryDcdcPublisher::subscribeToTemperatureChanged(VehiclePowertrainTractionBatteryDcdcTemperaturePropertyCb callback)
{
    auto handleId = m_temperatureChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_temperatureCallbacksMutex);
    m_temperatureCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryDcdcPublisher::unsubscribeFromTemperatureChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_temperatureCallbacksMutex);
    m_temperatureCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryDcdcPublisher::publishTemperatureChanged(float Temperature) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTemperatureChanged(Temperature);
    }
    std::shared_lock<std::shared_timed_mutex> temperatureCallbacksLock(m_temperatureCallbacksMutex);
    const auto temperatureCallbacks = m_temperatureCallbacks;
    temperatureCallbacksLock.unlock();
    for(const auto& callbackEntry: temperatureCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Temperature);
        }
    }
}

