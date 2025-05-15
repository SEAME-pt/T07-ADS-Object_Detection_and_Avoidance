

#include "vehicle/generated/core/vehiclepowertraintractionbattery.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehiclePowertrainTractionBatteryPublisher::subscribeToAllChanges(IVehiclePowertrainTractionBatterySubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehiclePowertrainTractionBatterySubscriber>(subscriber));
    }
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromAllChanges(IVehiclePowertrainTractionBatterySubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehiclePowertrainTractionBatteryPublisher::subscribeToIdChanged(VehiclePowertrainTractionBatteryIdPropertyCb callback)
{
    auto handleId = m_idChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_idCallbacksMutex);
    m_idCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromIdChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_idCallbacksMutex);
    m_idCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishIdChanged(const std::string& Id) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIdChanged(Id);
    }
    std::shared_lock<std::shared_timed_mutex> idCallbacksLock(m_idCallbacksMutex);
    const auto idCallbacks = m_idCallbacks;
    idCallbacksLock.unlock();
    for(const auto& callbackEntry: idCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Id);
        }
    }
}

long VehiclePowertrainTractionBatteryPublisher::subscribeToProductionDateChanged(VehiclePowertrainTractionBatteryProductionDatePropertyCb callback)
{
    auto handleId = m_productionDateChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_productionDateCallbacksMutex);
    m_productionDateCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromProductionDateChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_productionDateCallbacksMutex);
    m_productionDateCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishProductionDateChanged(const std::string& ProductionDate) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onProductionDateChanged(ProductionDate);
    }
    std::shared_lock<std::shared_timed_mutex> productionDateCallbacksLock(m_productionDateCallbacksMutex);
    const auto productionDateCallbacks = m_productionDateCallbacks;
    productionDateCallbacksLock.unlock();
    for(const auto& callbackEntry: productionDateCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(ProductionDate);
        }
    }
}

long VehiclePowertrainTractionBatteryPublisher::subscribeToErrorCodesChanged(VehiclePowertrainTractionBatteryErrorCodesPropertyCb callback)
{
    auto handleId = m_errorCodesChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_errorCodesCallbacksMutex);
    m_errorCodesCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromErrorCodesChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_errorCodesCallbacksMutex);
    m_errorCodesCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishErrorCodesChanged(const std::list<std::string>& ErrorCodes) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onErrorCodesChanged(ErrorCodes);
    }
    std::shared_lock<std::shared_timed_mutex> errorCodesCallbacksLock(m_errorCodesCallbacksMutex);
    const auto errorCodesCallbacks = m_errorCodesCallbacks;
    errorCodesCallbacksLock.unlock();
    for(const auto& callbackEntry: errorCodesCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(ErrorCodes);
        }
    }
}

long VehiclePowertrainTractionBatteryPublisher::subscribeToIsPowerConnectedChanged(VehiclePowertrainTractionBatteryIsPowerConnectedPropertyCb callback)
{
    auto handleId = m_isPowerConnectedChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isPowerConnectedCallbacksMutex);
    m_isPowerConnectedCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromIsPowerConnectedChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isPowerConnectedCallbacksMutex);
    m_isPowerConnectedCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishIsPowerConnectedChanged(bool IsPowerConnected) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsPowerConnectedChanged(IsPowerConnected);
    }
    std::shared_lock<std::shared_timed_mutex> isPowerConnectedCallbacksLock(m_isPowerConnectedCallbacksMutex);
    const auto isPowerConnectedCallbacks = m_isPowerConnectedCallbacks;
    isPowerConnectedCallbacksLock.unlock();
    for(const auto& callbackEntry: isPowerConnectedCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsPowerConnected);
        }
    }
}

long VehiclePowertrainTractionBatteryPublisher::subscribeToIsGroundConnectedChanged(VehiclePowertrainTractionBatteryIsGroundConnectedPropertyCb callback)
{
    auto handleId = m_isGroundConnectedChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isGroundConnectedCallbacksMutex);
    m_isGroundConnectedCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromIsGroundConnectedChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isGroundConnectedCallbacksMutex);
    m_isGroundConnectedCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishIsGroundConnectedChanged(bool IsGroundConnected) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsGroundConnectedChanged(IsGroundConnected);
    }
    std::shared_lock<std::shared_timed_mutex> isGroundConnectedCallbacksLock(m_isGroundConnectedCallbacksMutex);
    const auto isGroundConnectedCallbacks = m_isGroundConnectedCallbacks;
    isGroundConnectedCallbacksLock.unlock();
    for(const auto& callbackEntry: isGroundConnectedCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsGroundConnected);
        }
    }
}

long VehiclePowertrainTractionBatteryPublisher::subscribeToGrossCapacityChanged(VehiclePowertrainTractionBatteryGrossCapacityPropertyCb callback)
{
    auto handleId = m_grossCapacityChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_grossCapacityCallbacksMutex);
    m_grossCapacityCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromGrossCapacityChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_grossCapacityCallbacksMutex);
    m_grossCapacityCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishGrossCapacityChanged(int32_t GrossCapacity) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onGrossCapacityChanged(GrossCapacity);
    }
    std::shared_lock<std::shared_timed_mutex> grossCapacityCallbacksLock(m_grossCapacityCallbacksMutex);
    const auto grossCapacityCallbacks = m_grossCapacityCallbacks;
    grossCapacityCallbacksLock.unlock();
    for(const auto& callbackEntry: grossCapacityCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(GrossCapacity);
        }
    }
}

long VehiclePowertrainTractionBatteryPublisher::subscribeToNetCapacityChanged(VehiclePowertrainTractionBatteryNetCapacityPropertyCb callback)
{
    auto handleId = m_netCapacityChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_netCapacityCallbacksMutex);
    m_netCapacityCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromNetCapacityChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_netCapacityCallbacksMutex);
    m_netCapacityCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishNetCapacityChanged(int32_t NetCapacity) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onNetCapacityChanged(NetCapacity);
    }
    std::shared_lock<std::shared_timed_mutex> netCapacityCallbacksLock(m_netCapacityCallbacksMutex);
    const auto netCapacityCallbacks = m_netCapacityCallbacks;
    netCapacityCallbacksLock.unlock();
    for(const auto& callbackEntry: netCapacityCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(NetCapacity);
        }
    }
}

long VehiclePowertrainTractionBatteryPublisher::subscribeToStateOfHealthChanged(VehiclePowertrainTractionBatteryStateOfHealthPropertyCb callback)
{
    auto handleId = m_stateOfHealthChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_stateOfHealthCallbacksMutex);
    m_stateOfHealthCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromStateOfHealthChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_stateOfHealthCallbacksMutex);
    m_stateOfHealthCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishStateOfHealthChanged(float StateOfHealth) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onStateOfHealthChanged(StateOfHealth);
    }
    std::shared_lock<std::shared_timed_mutex> stateOfHealthCallbacksLock(m_stateOfHealthCallbacksMutex);
    const auto stateOfHealthCallbacks = m_stateOfHealthCallbacks;
    stateOfHealthCallbacksLock.unlock();
    for(const auto& callbackEntry: stateOfHealthCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(StateOfHealth);
        }
    }
}

long VehiclePowertrainTractionBatteryPublisher::subscribeToNominalVoltageChanged(VehiclePowertrainTractionBatteryNominalVoltagePropertyCb callback)
{
    auto handleId = m_nominalVoltageChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_nominalVoltageCallbacksMutex);
    m_nominalVoltageCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromNominalVoltageChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_nominalVoltageCallbacksMutex);
    m_nominalVoltageCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishNominalVoltageChanged(int32_t NominalVoltage) const
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

long VehiclePowertrainTractionBatteryPublisher::subscribeToMaxVoltageChanged(VehiclePowertrainTractionBatteryMaxVoltagePropertyCb callback)
{
    auto handleId = m_maxVoltageChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_maxVoltageCallbacksMutex);
    m_maxVoltageCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromMaxVoltageChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_maxVoltageCallbacksMutex);
    m_maxVoltageCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishMaxVoltageChanged(int32_t MaxVoltage) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onMaxVoltageChanged(MaxVoltage);
    }
    std::shared_lock<std::shared_timed_mutex> maxVoltageCallbacksLock(m_maxVoltageCallbacksMutex);
    const auto maxVoltageCallbacks = m_maxVoltageCallbacks;
    maxVoltageCallbacksLock.unlock();
    for(const auto& callbackEntry: maxVoltageCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(MaxVoltage);
        }
    }
}

long VehiclePowertrainTractionBatteryPublisher::subscribeToCurrentVoltageChanged(VehiclePowertrainTractionBatteryCurrentVoltagePropertyCb callback)
{
    auto handleId = m_currentVoltageChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_currentVoltageCallbacksMutex);
    m_currentVoltageCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromCurrentVoltageChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_currentVoltageCallbacksMutex);
    m_currentVoltageCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishCurrentVoltageChanged(float CurrentVoltage) const
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

long VehiclePowertrainTractionBatteryPublisher::subscribeToCurrentCurrentChanged(VehiclePowertrainTractionBatteryCurrentCurrentPropertyCb callback)
{
    auto handleId = m_currentCurrentChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_currentCurrentCallbacksMutex);
    m_currentCurrentCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromCurrentCurrentChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_currentCurrentCallbacksMutex);
    m_currentCurrentCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishCurrentCurrentChanged(float CurrentCurrent) const
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

long VehiclePowertrainTractionBatteryPublisher::subscribeToCurrentPowerChanged(VehiclePowertrainTractionBatteryCurrentPowerPropertyCb callback)
{
    auto handleId = m_currentPowerChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_currentPowerCallbacksMutex);
    m_currentPowerCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromCurrentPowerChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_currentPowerCallbacksMutex);
    m_currentPowerCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishCurrentPowerChanged(float CurrentPower) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onCurrentPowerChanged(CurrentPower);
    }
    std::shared_lock<std::shared_timed_mutex> currentPowerCallbacksLock(m_currentPowerCallbacksMutex);
    const auto currentPowerCallbacks = m_currentPowerCallbacks;
    currentPowerCallbacksLock.unlock();
    for(const auto& callbackEntry: currentPowerCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(CurrentPower);
        }
    }
}

long VehiclePowertrainTractionBatteryPublisher::subscribeToAccumulatedChargedEnergyChanged(VehiclePowertrainTractionBatteryAccumulatedChargedEnergyPropertyCb callback)
{
    auto handleId = m_accumulatedChargedEnergyChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_accumulatedChargedEnergyCallbacksMutex);
    m_accumulatedChargedEnergyCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromAccumulatedChargedEnergyChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_accumulatedChargedEnergyCallbacksMutex);
    m_accumulatedChargedEnergyCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishAccumulatedChargedEnergyChanged(float AccumulatedChargedEnergy) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onAccumulatedChargedEnergyChanged(AccumulatedChargedEnergy);
    }
    std::shared_lock<std::shared_timed_mutex> accumulatedChargedEnergyCallbacksLock(m_accumulatedChargedEnergyCallbacksMutex);
    const auto accumulatedChargedEnergyCallbacks = m_accumulatedChargedEnergyCallbacks;
    accumulatedChargedEnergyCallbacksLock.unlock();
    for(const auto& callbackEntry: accumulatedChargedEnergyCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(AccumulatedChargedEnergy);
        }
    }
}

long VehiclePowertrainTractionBatteryPublisher::subscribeToAccumulatedConsumedEnergyChanged(VehiclePowertrainTractionBatteryAccumulatedConsumedEnergyPropertyCb callback)
{
    auto handleId = m_accumulatedConsumedEnergyChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_accumulatedConsumedEnergyCallbacksMutex);
    m_accumulatedConsumedEnergyCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromAccumulatedConsumedEnergyChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_accumulatedConsumedEnergyCallbacksMutex);
    m_accumulatedConsumedEnergyCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishAccumulatedConsumedEnergyChanged(float AccumulatedConsumedEnergy) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onAccumulatedConsumedEnergyChanged(AccumulatedConsumedEnergy);
    }
    std::shared_lock<std::shared_timed_mutex> accumulatedConsumedEnergyCallbacksLock(m_accumulatedConsumedEnergyCallbacksMutex);
    const auto accumulatedConsumedEnergyCallbacks = m_accumulatedConsumedEnergyCallbacks;
    accumulatedConsumedEnergyCallbacksLock.unlock();
    for(const auto& callbackEntry: accumulatedConsumedEnergyCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(AccumulatedConsumedEnergy);
        }
    }
}

long VehiclePowertrainTractionBatteryPublisher::subscribeToAccumulatedChargedThroughputChanged(VehiclePowertrainTractionBatteryAccumulatedChargedThroughputPropertyCb callback)
{
    auto handleId = m_accumulatedChargedThroughputChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_accumulatedChargedThroughputCallbacksMutex);
    m_accumulatedChargedThroughputCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromAccumulatedChargedThroughputChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_accumulatedChargedThroughputCallbacksMutex);
    m_accumulatedChargedThroughputCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishAccumulatedChargedThroughputChanged(float AccumulatedChargedThroughput) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onAccumulatedChargedThroughputChanged(AccumulatedChargedThroughput);
    }
    std::shared_lock<std::shared_timed_mutex> accumulatedChargedThroughputCallbacksLock(m_accumulatedChargedThroughputCallbacksMutex);
    const auto accumulatedChargedThroughputCallbacks = m_accumulatedChargedThroughputCallbacks;
    accumulatedChargedThroughputCallbacksLock.unlock();
    for(const auto& callbackEntry: accumulatedChargedThroughputCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(AccumulatedChargedThroughput);
        }
    }
}

long VehiclePowertrainTractionBatteryPublisher::subscribeToAccumulatedConsumedThroughputChanged(VehiclePowertrainTractionBatteryAccumulatedConsumedThroughputPropertyCb callback)
{
    auto handleId = m_accumulatedConsumedThroughputChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_accumulatedConsumedThroughputCallbacksMutex);
    m_accumulatedConsumedThroughputCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromAccumulatedConsumedThroughputChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_accumulatedConsumedThroughputCallbacksMutex);
    m_accumulatedConsumedThroughputCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishAccumulatedConsumedThroughputChanged(float AccumulatedConsumedThroughput) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onAccumulatedConsumedThroughputChanged(AccumulatedConsumedThroughput);
    }
    std::shared_lock<std::shared_timed_mutex> accumulatedConsumedThroughputCallbacksLock(m_accumulatedConsumedThroughputCallbacksMutex);
    const auto accumulatedConsumedThroughputCallbacks = m_accumulatedConsumedThroughputCallbacks;
    accumulatedConsumedThroughputCallbacksLock.unlock();
    for(const auto& callbackEntry: accumulatedConsumedThroughputCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(AccumulatedConsumedThroughput);
        }
    }
}

long VehiclePowertrainTractionBatteryPublisher::subscribeToPowerLossChanged(VehiclePowertrainTractionBatteryPowerLossPropertyCb callback)
{
    auto handleId = m_powerLossChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_powerLossCallbacksMutex);
    m_powerLossCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromPowerLossChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_powerLossCallbacksMutex);
    m_powerLossCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishPowerLossChanged(float PowerLoss) const
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

long VehiclePowertrainTractionBatteryPublisher::subscribeToRangeChanged(VehiclePowertrainTractionBatteryRangePropertyCb callback)
{
    auto handleId = m_rangeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_rangeCallbacksMutex);
    m_rangeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromRangeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rangeCallbacksMutex);
    m_rangeCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishRangeChanged(int32_t Range) const
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

long VehiclePowertrainTractionBatteryPublisher::subscribeToTimeRemainingChanged(VehiclePowertrainTractionBatteryTimeRemainingPropertyCb callback)
{
    auto handleId = m_timeRemainingChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_timeRemainingCallbacksMutex);
    m_timeRemainingCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTractionBatteryPublisher::unsubscribeFromTimeRemainingChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_timeRemainingCallbacksMutex);
    m_timeRemainingCallbacks.erase(handleId);
}

void VehiclePowertrainTractionBatteryPublisher::publishTimeRemainingChanged(int32_t TimeRemaining) const
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

