

#include "vehicle/generated/core/vehiclepowertrainelectricmotorenginecoolant.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehiclePowertrainElectricMotorEngineCoolantPublisher::subscribeToAllChanges(IVehiclePowertrainElectricMotorEngineCoolantSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehiclePowertrainElectricMotorEngineCoolantSubscriber>(subscriber));
    }
}

void VehiclePowertrainElectricMotorEngineCoolantPublisher::unsubscribeFromAllChanges(IVehiclePowertrainElectricMotorEngineCoolantSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehiclePowertrainElectricMotorEngineCoolantPublisher::subscribeToCapacityChanged(VehiclePowertrainElectricMotorEngineCoolantCapacityPropertyCb callback)
{
    auto handleId = m_capacityChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_capacityCallbacksMutex);
    m_capacityCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainElectricMotorEngineCoolantPublisher::unsubscribeFromCapacityChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_capacityCallbacksMutex);
    m_capacityCallbacks.erase(handleId);
}

void VehiclePowertrainElectricMotorEngineCoolantPublisher::publishCapacityChanged(float Capacity) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onCapacityChanged(Capacity);
    }
    std::shared_lock<std::shared_timed_mutex> capacityCallbacksLock(m_capacityCallbacksMutex);
    const auto capacityCallbacks = m_capacityCallbacks;
    capacityCallbacksLock.unlock();
    for(const auto& callbackEntry: capacityCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Capacity);
        }
    }
}

long VehiclePowertrainElectricMotorEngineCoolantPublisher::subscribeToTemperatureChanged(VehiclePowertrainElectricMotorEngineCoolantTemperaturePropertyCb callback)
{
    auto handleId = m_temperatureChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_temperatureCallbacksMutex);
    m_temperatureCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainElectricMotorEngineCoolantPublisher::unsubscribeFromTemperatureChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_temperatureCallbacksMutex);
    m_temperatureCallbacks.erase(handleId);
}

void VehiclePowertrainElectricMotorEngineCoolantPublisher::publishTemperatureChanged(float Temperature) const
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

long VehiclePowertrainElectricMotorEngineCoolantPublisher::subscribeToLevelChanged(VehiclePowertrainElectricMotorEngineCoolantLevelPropertyCb callback)
{
    auto handleId = m_levelChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_levelCallbacksMutex);
    m_levelCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainElectricMotorEngineCoolantPublisher::unsubscribeFromLevelChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_levelCallbacksMutex);
    m_levelCallbacks.erase(handleId);
}

void VehiclePowertrainElectricMotorEngineCoolantPublisher::publishLevelChanged(Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum Level) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onLevelChanged(Level);
    }
    std::shared_lock<std::shared_timed_mutex> levelCallbacksLock(m_levelCallbacksMutex);
    const auto levelCallbacks = m_levelCallbacks;
    levelCallbacksLock.unlock();
    for(const auto& callbackEntry: levelCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Level);
        }
    }
}

long VehiclePowertrainElectricMotorEngineCoolantPublisher::subscribeToLifeRemainingChanged(VehiclePowertrainElectricMotorEngineCoolantLifeRemainingPropertyCb callback)
{
    auto handleId = m_lifeRemainingChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_lifeRemainingCallbacksMutex);
    m_lifeRemainingCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainElectricMotorEngineCoolantPublisher::unsubscribeFromLifeRemainingChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_lifeRemainingCallbacksMutex);
    m_lifeRemainingCallbacks.erase(handleId);
}

void VehiclePowertrainElectricMotorEngineCoolantPublisher::publishLifeRemainingChanged(int32_t LifeRemaining) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onLifeRemainingChanged(LifeRemaining);
    }
    std::shared_lock<std::shared_timed_mutex> lifeRemainingCallbacksLock(m_lifeRemainingCallbacksMutex);
    const auto lifeRemainingCallbacks = m_lifeRemainingCallbacks;
    lifeRemainingCallbacksLock.unlock();
    for(const auto& callbackEntry: lifeRemainingCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(LifeRemaining);
        }
    }
}

