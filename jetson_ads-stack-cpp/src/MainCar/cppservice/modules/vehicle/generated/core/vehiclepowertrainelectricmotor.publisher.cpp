

#include "vehicle/generated/core/vehiclepowertrainelectricmotor.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehiclePowertrainElectricMotorPublisher::subscribeToAllChanges(IVehiclePowertrainElectricMotorSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehiclePowertrainElectricMotorSubscriber>(subscriber));
    }
}

void VehiclePowertrainElectricMotorPublisher::unsubscribeFromAllChanges(IVehiclePowertrainElectricMotorSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehiclePowertrainElectricMotorPublisher::subscribeToEngineCodeChanged(VehiclePowertrainElectricMotorEngineCodePropertyCb callback)
{
    auto handleId = m_engineCodeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_engineCodeCallbacksMutex);
    m_engineCodeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainElectricMotorPublisher::unsubscribeFromEngineCodeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_engineCodeCallbacksMutex);
    m_engineCodeCallbacks.erase(handleId);
}

void VehiclePowertrainElectricMotorPublisher::publishEngineCodeChanged(const std::string& EngineCode) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onEngineCodeChanged(EngineCode);
    }
    std::shared_lock<std::shared_timed_mutex> engineCodeCallbacksLock(m_engineCodeCallbacksMutex);
    const auto engineCodeCallbacks = m_engineCodeCallbacks;
    engineCodeCallbacksLock.unlock();
    for(const auto& callbackEntry: engineCodeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(EngineCode);
        }
    }
}

long VehiclePowertrainElectricMotorPublisher::subscribeToMaxPowerChanged(VehiclePowertrainElectricMotorMaxPowerPropertyCb callback)
{
    auto handleId = m_maxPowerChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_maxPowerCallbacksMutex);
    m_maxPowerCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainElectricMotorPublisher::unsubscribeFromMaxPowerChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_maxPowerCallbacksMutex);
    m_maxPowerCallbacks.erase(handleId);
}

void VehiclePowertrainElectricMotorPublisher::publishMaxPowerChanged(int32_t MaxPower) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onMaxPowerChanged(MaxPower);
    }
    std::shared_lock<std::shared_timed_mutex> maxPowerCallbacksLock(m_maxPowerCallbacksMutex);
    const auto maxPowerCallbacks = m_maxPowerCallbacks;
    maxPowerCallbacksLock.unlock();
    for(const auto& callbackEntry: maxPowerCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(MaxPower);
        }
    }
}

long VehiclePowertrainElectricMotorPublisher::subscribeToMaxTorqueChanged(VehiclePowertrainElectricMotorMaxTorquePropertyCb callback)
{
    auto handleId = m_maxTorqueChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_maxTorqueCallbacksMutex);
    m_maxTorqueCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainElectricMotorPublisher::unsubscribeFromMaxTorqueChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_maxTorqueCallbacksMutex);
    m_maxTorqueCallbacks.erase(handleId);
}

void VehiclePowertrainElectricMotorPublisher::publishMaxTorqueChanged(int32_t MaxTorque) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onMaxTorqueChanged(MaxTorque);
    }
    std::shared_lock<std::shared_timed_mutex> maxTorqueCallbacksLock(m_maxTorqueCallbacksMutex);
    const auto maxTorqueCallbacks = m_maxTorqueCallbacks;
    maxTorqueCallbacksLock.unlock();
    for(const auto& callbackEntry: maxTorqueCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(MaxTorque);
        }
    }
}

long VehiclePowertrainElectricMotorPublisher::subscribeToMaxRegenPowerChanged(VehiclePowertrainElectricMotorMaxRegenPowerPropertyCb callback)
{
    auto handleId = m_maxRegenPowerChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_maxRegenPowerCallbacksMutex);
    m_maxRegenPowerCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainElectricMotorPublisher::unsubscribeFromMaxRegenPowerChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_maxRegenPowerCallbacksMutex);
    m_maxRegenPowerCallbacks.erase(handleId);
}

void VehiclePowertrainElectricMotorPublisher::publishMaxRegenPowerChanged(int32_t MaxRegenPower) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onMaxRegenPowerChanged(MaxRegenPower);
    }
    std::shared_lock<std::shared_timed_mutex> maxRegenPowerCallbacksLock(m_maxRegenPowerCallbacksMutex);
    const auto maxRegenPowerCallbacks = m_maxRegenPowerCallbacks;
    maxRegenPowerCallbacksLock.unlock();
    for(const auto& callbackEntry: maxRegenPowerCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(MaxRegenPower);
        }
    }
}

long VehiclePowertrainElectricMotorPublisher::subscribeToMaxRegenTorqueChanged(VehiclePowertrainElectricMotorMaxRegenTorquePropertyCb callback)
{
    auto handleId = m_maxRegenTorqueChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_maxRegenTorqueCallbacksMutex);
    m_maxRegenTorqueCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainElectricMotorPublisher::unsubscribeFromMaxRegenTorqueChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_maxRegenTorqueCallbacksMutex);
    m_maxRegenTorqueCallbacks.erase(handleId);
}

void VehiclePowertrainElectricMotorPublisher::publishMaxRegenTorqueChanged(int32_t MaxRegenTorque) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onMaxRegenTorqueChanged(MaxRegenTorque);
    }
    std::shared_lock<std::shared_timed_mutex> maxRegenTorqueCallbacksLock(m_maxRegenTorqueCallbacksMutex);
    const auto maxRegenTorqueCallbacks = m_maxRegenTorqueCallbacks;
    maxRegenTorqueCallbacksLock.unlock();
    for(const auto& callbackEntry: maxRegenTorqueCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(MaxRegenTorque);
        }
    }
}

long VehiclePowertrainElectricMotorPublisher::subscribeToSpeedChanged(VehiclePowertrainElectricMotorSpeedPropertyCb callback)
{
    auto handleId = m_speedChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_speedCallbacksMutex);
    m_speedCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainElectricMotorPublisher::unsubscribeFromSpeedChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_speedCallbacksMutex);
    m_speedCallbacks.erase(handleId);
}

void VehiclePowertrainElectricMotorPublisher::publishSpeedChanged(int32_t Speed) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onSpeedChanged(Speed);
    }
    std::shared_lock<std::shared_timed_mutex> speedCallbacksLock(m_speedCallbacksMutex);
    const auto speedCallbacks = m_speedCallbacks;
    speedCallbacksLock.unlock();
    for(const auto& callbackEntry: speedCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Speed);
        }
    }
}

long VehiclePowertrainElectricMotorPublisher::subscribeToTimeInUseChanged(VehiclePowertrainElectricMotorTimeInUsePropertyCb callback)
{
    auto handleId = m_timeInUseChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_timeInUseCallbacksMutex);
    m_timeInUseCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainElectricMotorPublisher::unsubscribeFromTimeInUseChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_timeInUseCallbacksMutex);
    m_timeInUseCallbacks.erase(handleId);
}

void VehiclePowertrainElectricMotorPublisher::publishTimeInUseChanged(float TimeInUse) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTimeInUseChanged(TimeInUse);
    }
    std::shared_lock<std::shared_timed_mutex> timeInUseCallbacksLock(m_timeInUseCallbacksMutex);
    const auto timeInUseCallbacks = m_timeInUseCallbacks;
    timeInUseCallbacksLock.unlock();
    for(const auto& callbackEntry: timeInUseCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TimeInUse);
        }
    }
}

long VehiclePowertrainElectricMotorPublisher::subscribeToTemperatureChanged(VehiclePowertrainElectricMotorTemperaturePropertyCb callback)
{
    auto handleId = m_temperatureChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_temperatureCallbacksMutex);
    m_temperatureCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainElectricMotorPublisher::unsubscribeFromTemperatureChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_temperatureCallbacksMutex);
    m_temperatureCallbacks.erase(handleId);
}

void VehiclePowertrainElectricMotorPublisher::publishTemperatureChanged(float Temperature) const
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

long VehiclePowertrainElectricMotorPublisher::subscribeToCoolantTemperatureDeprecatedChanged(VehiclePowertrainElectricMotorCoolantTemperatureDeprecatedPropertyCb callback)
{
    auto handleId = m_coolantTemperatureDeprecatedChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_coolantTemperatureDeprecatedCallbacksMutex);
    m_coolantTemperatureDeprecatedCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainElectricMotorPublisher::unsubscribeFromCoolantTemperatureDeprecatedChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_coolantTemperatureDeprecatedCallbacksMutex);
    m_coolantTemperatureDeprecatedCallbacks.erase(handleId);
}

void VehiclePowertrainElectricMotorPublisher::publishCoolantTemperatureDeprecatedChanged(float CoolantTemperature_Deprecated) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onCoolantTemperatureDeprecatedChanged(CoolantTemperature_Deprecated);
    }
    std::shared_lock<std::shared_timed_mutex> coolantTemperatureDeprecatedCallbacksLock(m_coolantTemperatureDeprecatedCallbacksMutex);
    const auto coolantTemperatureDeprecatedCallbacks = m_coolantTemperatureDeprecatedCallbacks;
    coolantTemperatureDeprecatedCallbacksLock.unlock();
    for(const auto& callbackEntry: coolantTemperatureDeprecatedCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(CoolantTemperature_Deprecated);
        }
    }
}

long VehiclePowertrainElectricMotorPublisher::subscribeToPowerChanged(VehiclePowertrainElectricMotorPowerPropertyCb callback)
{
    auto handleId = m_powerChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_powerCallbacksMutex);
    m_powerCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainElectricMotorPublisher::unsubscribeFromPowerChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_powerCallbacksMutex);
    m_powerCallbacks.erase(handleId);
}

void VehiclePowertrainElectricMotorPublisher::publishPowerChanged(int32_t Power) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onPowerChanged(Power);
    }
    std::shared_lock<std::shared_timed_mutex> powerCallbacksLock(m_powerCallbacksMutex);
    const auto powerCallbacks = m_powerCallbacks;
    powerCallbacksLock.unlock();
    for(const auto& callbackEntry: powerCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Power);
        }
    }
}

long VehiclePowertrainElectricMotorPublisher::subscribeToTorqueChanged(VehiclePowertrainElectricMotorTorquePropertyCb callback)
{
    auto handleId = m_torqueChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueCallbacksMutex);
    m_torqueCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainElectricMotorPublisher::unsubscribeFromTorqueChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueCallbacksMutex);
    m_torqueCallbacks.erase(handleId);
}

void VehiclePowertrainElectricMotorPublisher::publishTorqueChanged(int32_t Torque) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueChanged(Torque);
    }
    std::shared_lock<std::shared_timed_mutex> torqueCallbacksLock(m_torqueCallbacksMutex);
    const auto torqueCallbacks = m_torqueCallbacks;
    torqueCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Torque);
        }
    }
}

