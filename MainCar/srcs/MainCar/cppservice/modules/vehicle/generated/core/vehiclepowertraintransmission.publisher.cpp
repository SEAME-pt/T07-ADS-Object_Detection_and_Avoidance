

#include "vehicle/generated/core/vehiclepowertraintransmission.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehiclePowertrainTransmissionPublisher::subscribeToAllChanges(IVehiclePowertrainTransmissionSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehiclePowertrainTransmissionSubscriber>(subscriber));
    }
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromAllChanges(IVehiclePowertrainTransmissionSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehiclePowertrainTransmissionPublisher::subscribeToTypeChanged(VehiclePowertrainTransmissionTypePropertyCb callback)
{
    auto handleId = m_typeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_typeCallbacksMutex);
    m_typeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromTypeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_typeCallbacksMutex);
    m_typeCallbacks.erase(handleId);
}

void VehiclePowertrainTransmissionPublisher::publishTypeChanged(Vehicle_Powertrain_Transmission_Type_ValueEnum Type) const
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

long VehiclePowertrainTransmissionPublisher::subscribeToGearCountChanged(VehiclePowertrainTransmissionGearCountPropertyCb callback)
{
    auto handleId = m_gearCountChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_gearCountCallbacksMutex);
    m_gearCountCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromGearCountChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_gearCountCallbacksMutex);
    m_gearCountCallbacks.erase(handleId);
}

void VehiclePowertrainTransmissionPublisher::publishGearCountChanged(int32_t GearCount) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onGearCountChanged(GearCount);
    }
    std::shared_lock<std::shared_timed_mutex> gearCountCallbacksLock(m_gearCountCallbacksMutex);
    const auto gearCountCallbacks = m_gearCountCallbacks;
    gearCountCallbacksLock.unlock();
    for(const auto& callbackEntry: gearCountCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(GearCount);
        }
    }
}

long VehiclePowertrainTransmissionPublisher::subscribeToDriveTypeChanged(VehiclePowertrainTransmissionDriveTypePropertyCb callback)
{
    auto handleId = m_driveTypeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_driveTypeCallbacksMutex);
    m_driveTypeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromDriveTypeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_driveTypeCallbacksMutex);
    m_driveTypeCallbacks.erase(handleId);
}

void VehiclePowertrainTransmissionPublisher::publishDriveTypeChanged(Vehicle_Powertrain_Transmission_DriveType_ValueEnum DriveType) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onDriveTypeChanged(DriveType);
    }
    std::shared_lock<std::shared_timed_mutex> driveTypeCallbacksLock(m_driveTypeCallbacksMutex);
    const auto driveTypeCallbacks = m_driveTypeCallbacks;
    driveTypeCallbacksLock.unlock();
    for(const auto& callbackEntry: driveTypeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(DriveType);
        }
    }
}

long VehiclePowertrainTransmissionPublisher::subscribeToTravelledDistanceChanged(VehiclePowertrainTransmissionTravelledDistancePropertyCb callback)
{
    auto handleId = m_travelledDistanceChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_travelledDistanceCallbacksMutex);
    m_travelledDistanceCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromTravelledDistanceChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_travelledDistanceCallbacksMutex);
    m_travelledDistanceCallbacks.erase(handleId);
}

void VehiclePowertrainTransmissionPublisher::publishTravelledDistanceChanged(float TravelledDistance) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTravelledDistanceChanged(TravelledDistance);
    }
    std::shared_lock<std::shared_timed_mutex> travelledDistanceCallbacksLock(m_travelledDistanceCallbacksMutex);
    const auto travelledDistanceCallbacks = m_travelledDistanceCallbacks;
    travelledDistanceCallbacksLock.unlock();
    for(const auto& callbackEntry: travelledDistanceCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TravelledDistance);
        }
    }
}

long VehiclePowertrainTransmissionPublisher::subscribeToCurrentGearChanged(VehiclePowertrainTransmissionCurrentGearPropertyCb callback)
{
    auto handleId = m_currentGearChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_currentGearCallbacksMutex);
    m_currentGearCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromCurrentGearChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_currentGearCallbacksMutex);
    m_currentGearCallbacks.erase(handleId);
}

void VehiclePowertrainTransmissionPublisher::publishCurrentGearChanged(int32_t CurrentGear) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onCurrentGearChanged(CurrentGear);
    }
    std::shared_lock<std::shared_timed_mutex> currentGearCallbacksLock(m_currentGearCallbacksMutex);
    const auto currentGearCallbacks = m_currentGearCallbacks;
    currentGearCallbacksLock.unlock();
    for(const auto& callbackEntry: currentGearCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(CurrentGear);
        }
    }
}

long VehiclePowertrainTransmissionPublisher::subscribeToSelectedGearChanged(VehiclePowertrainTransmissionSelectedGearPropertyCb callback)
{
    auto handleId = m_selectedGearChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_selectedGearCallbacksMutex);
    m_selectedGearCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromSelectedGearChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_selectedGearCallbacksMutex);
    m_selectedGearCallbacks.erase(handleId);
}

void VehiclePowertrainTransmissionPublisher::publishSelectedGearChanged(int32_t SelectedGear) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onSelectedGearChanged(SelectedGear);
    }
    std::shared_lock<std::shared_timed_mutex> selectedGearCallbacksLock(m_selectedGearCallbacksMutex);
    const auto selectedGearCallbacks = m_selectedGearCallbacks;
    selectedGearCallbacksLock.unlock();
    for(const auto& callbackEntry: selectedGearCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(SelectedGear);
        }
    }
}

long VehiclePowertrainTransmissionPublisher::subscribeToIsParkLockEngagedChanged(VehiclePowertrainTransmissionIsParkLockEngagedPropertyCb callback)
{
    auto handleId = m_isParkLockEngagedChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isParkLockEngagedCallbacksMutex);
    m_isParkLockEngagedCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromIsParkLockEngagedChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isParkLockEngagedCallbacksMutex);
    m_isParkLockEngagedCallbacks.erase(handleId);
}

void VehiclePowertrainTransmissionPublisher::publishIsParkLockEngagedChanged(bool IsParkLockEngaged) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsParkLockEngagedChanged(IsParkLockEngaged);
    }
    std::shared_lock<std::shared_timed_mutex> isParkLockEngagedCallbacksLock(m_isParkLockEngagedCallbacksMutex);
    const auto isParkLockEngagedCallbacks = m_isParkLockEngagedCallbacks;
    isParkLockEngagedCallbacksLock.unlock();
    for(const auto& callbackEntry: isParkLockEngagedCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsParkLockEngaged);
        }
    }
}

long VehiclePowertrainTransmissionPublisher::subscribeToIsLowRangeEngagedChanged(VehiclePowertrainTransmissionIsLowRangeEngagedPropertyCb callback)
{
    auto handleId = m_isLowRangeEngagedChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isLowRangeEngagedCallbacksMutex);
    m_isLowRangeEngagedCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromIsLowRangeEngagedChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isLowRangeEngagedCallbacksMutex);
    m_isLowRangeEngagedCallbacks.erase(handleId);
}

void VehiclePowertrainTransmissionPublisher::publishIsLowRangeEngagedChanged(bool IsLowRangeEngaged) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsLowRangeEngagedChanged(IsLowRangeEngaged);
    }
    std::shared_lock<std::shared_timed_mutex> isLowRangeEngagedCallbacksLock(m_isLowRangeEngagedCallbacksMutex);
    const auto isLowRangeEngagedCallbacks = m_isLowRangeEngagedCallbacks;
    isLowRangeEngagedCallbacksLock.unlock();
    for(const auto& callbackEntry: isLowRangeEngagedCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsLowRangeEngaged);
        }
    }
}

long VehiclePowertrainTransmissionPublisher::subscribeToIsElectricalPowertrainEngagedChanged(VehiclePowertrainTransmissionIsElectricalPowertrainEngagedPropertyCb callback)
{
    auto handleId = m_isElectricalPowertrainEngagedChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isElectricalPowertrainEngagedCallbacksMutex);
    m_isElectricalPowertrainEngagedCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromIsElectricalPowertrainEngagedChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isElectricalPowertrainEngagedCallbacksMutex);
    m_isElectricalPowertrainEngagedCallbacks.erase(handleId);
}

void VehiclePowertrainTransmissionPublisher::publishIsElectricalPowertrainEngagedChanged(bool IsElectricalPowertrainEngaged) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsElectricalPowertrainEngagedChanged(IsElectricalPowertrainEngaged);
    }
    std::shared_lock<std::shared_timed_mutex> isElectricalPowertrainEngagedCallbacksLock(m_isElectricalPowertrainEngagedCallbacksMutex);
    const auto isElectricalPowertrainEngagedCallbacks = m_isElectricalPowertrainEngagedCallbacks;
    isElectricalPowertrainEngagedCallbacksLock.unlock();
    for(const auto& callbackEntry: isElectricalPowertrainEngagedCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsElectricalPowertrainEngaged);
        }
    }
}

long VehiclePowertrainTransmissionPublisher::subscribeToPerformanceModeChanged(VehiclePowertrainTransmissionPerformanceModePropertyCb callback)
{
    auto handleId = m_performanceModeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_performanceModeCallbacksMutex);
    m_performanceModeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromPerformanceModeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_performanceModeCallbacksMutex);
    m_performanceModeCallbacks.erase(handleId);
}

void VehiclePowertrainTransmissionPublisher::publishPerformanceModeChanged(Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum PerformanceMode) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onPerformanceModeChanged(PerformanceMode);
    }
    std::shared_lock<std::shared_timed_mutex> performanceModeCallbacksLock(m_performanceModeCallbacksMutex);
    const auto performanceModeCallbacks = m_performanceModeCallbacks;
    performanceModeCallbacksLock.unlock();
    for(const auto& callbackEntry: performanceModeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(PerformanceMode);
        }
    }
}

long VehiclePowertrainTransmissionPublisher::subscribeToGearChangeModeChanged(VehiclePowertrainTransmissionGearChangeModePropertyCb callback)
{
    auto handleId = m_gearChangeModeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_gearChangeModeCallbacksMutex);
    m_gearChangeModeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromGearChangeModeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_gearChangeModeCallbacksMutex);
    m_gearChangeModeCallbacks.erase(handleId);
}

void VehiclePowertrainTransmissionPublisher::publishGearChangeModeChanged(Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum GearChangeMode) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onGearChangeModeChanged(GearChangeMode);
    }
    std::shared_lock<std::shared_timed_mutex> gearChangeModeCallbacksLock(m_gearChangeModeCallbacksMutex);
    const auto gearChangeModeCallbacks = m_gearChangeModeCallbacks;
    gearChangeModeCallbacksLock.unlock();
    for(const auto& callbackEntry: gearChangeModeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(GearChangeMode);
        }
    }
}

long VehiclePowertrainTransmissionPublisher::subscribeToTemperatureChanged(VehiclePowertrainTransmissionTemperaturePropertyCb callback)
{
    auto handleId = m_temperatureChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_temperatureCallbacksMutex);
    m_temperatureCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromTemperatureChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_temperatureCallbacksMutex);
    m_temperatureCallbacks.erase(handleId);
}

void VehiclePowertrainTransmissionPublisher::publishTemperatureChanged(float Temperature) const
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

long VehiclePowertrainTransmissionPublisher::subscribeToClutchEngagementChanged(VehiclePowertrainTransmissionClutchEngagementPropertyCb callback)
{
    auto handleId = m_clutchEngagementChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_clutchEngagementCallbacksMutex);
    m_clutchEngagementCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromClutchEngagementChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_clutchEngagementCallbacksMutex);
    m_clutchEngagementCallbacks.erase(handleId);
}

void VehiclePowertrainTransmissionPublisher::publishClutchEngagementChanged(float ClutchEngagement) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onClutchEngagementChanged(ClutchEngagement);
    }
    std::shared_lock<std::shared_timed_mutex> clutchEngagementCallbacksLock(m_clutchEngagementCallbacksMutex);
    const auto clutchEngagementCallbacks = m_clutchEngagementCallbacks;
    clutchEngagementCallbacksLock.unlock();
    for(const auto& callbackEntry: clutchEngagementCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(ClutchEngagement);
        }
    }
}

long VehiclePowertrainTransmissionPublisher::subscribeToClutchWearChanged(VehiclePowertrainTransmissionClutchWearPropertyCb callback)
{
    auto handleId = m_clutchWearChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_clutchWearCallbacksMutex);
    m_clutchWearCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromClutchWearChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_clutchWearCallbacksMutex);
    m_clutchWearCallbacks.erase(handleId);
}

void VehiclePowertrainTransmissionPublisher::publishClutchWearChanged(int32_t ClutchWear) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onClutchWearChanged(ClutchWear);
    }
    std::shared_lock<std::shared_timed_mutex> clutchWearCallbacksLock(m_clutchWearCallbacksMutex);
    const auto clutchWearCallbacks = m_clutchWearCallbacks;
    clutchWearCallbacksLock.unlock();
    for(const auto& callbackEntry: clutchWearCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(ClutchWear);
        }
    }
}

long VehiclePowertrainTransmissionPublisher::subscribeToDiffLockFrontEngagementChanged(VehiclePowertrainTransmissionDiffLockFrontEngagementPropertyCb callback)
{
    auto handleId = m_diffLockFrontEngagementChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_diffLockFrontEngagementCallbacksMutex);
    m_diffLockFrontEngagementCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromDiffLockFrontEngagementChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_diffLockFrontEngagementCallbacksMutex);
    m_diffLockFrontEngagementCallbacks.erase(handleId);
}

void VehiclePowertrainTransmissionPublisher::publishDiffLockFrontEngagementChanged(float DiffLockFrontEngagement) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onDiffLockFrontEngagementChanged(DiffLockFrontEngagement);
    }
    std::shared_lock<std::shared_timed_mutex> diffLockFrontEngagementCallbacksLock(m_diffLockFrontEngagementCallbacksMutex);
    const auto diffLockFrontEngagementCallbacks = m_diffLockFrontEngagementCallbacks;
    diffLockFrontEngagementCallbacksLock.unlock();
    for(const auto& callbackEntry: diffLockFrontEngagementCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(DiffLockFrontEngagement);
        }
    }
}

long VehiclePowertrainTransmissionPublisher::subscribeToDiffLockRearEngagementChanged(VehiclePowertrainTransmissionDiffLockRearEngagementPropertyCb callback)
{
    auto handleId = m_diffLockRearEngagementChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_diffLockRearEngagementCallbacksMutex);
    m_diffLockRearEngagementCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromDiffLockRearEngagementChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_diffLockRearEngagementCallbacksMutex);
    m_diffLockRearEngagementCallbacks.erase(handleId);
}

void VehiclePowertrainTransmissionPublisher::publishDiffLockRearEngagementChanged(float DiffLockRearEngagement) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onDiffLockRearEngagementChanged(DiffLockRearEngagement);
    }
    std::shared_lock<std::shared_timed_mutex> diffLockRearEngagementCallbacksLock(m_diffLockRearEngagementCallbacksMutex);
    const auto diffLockRearEngagementCallbacks = m_diffLockRearEngagementCallbacks;
    diffLockRearEngagementCallbacksLock.unlock();
    for(const auto& callbackEntry: diffLockRearEngagementCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(DiffLockRearEngagement);
        }
    }
}

long VehiclePowertrainTransmissionPublisher::subscribeToTorqueDistributionChanged(VehiclePowertrainTransmissionTorqueDistributionPropertyCb callback)
{
    auto handleId = m_torqueDistributionChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueDistributionCallbacksMutex);
    m_torqueDistributionCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePowertrainTransmissionPublisher::unsubscribeFromTorqueDistributionChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueDistributionCallbacksMutex);
    m_torqueDistributionCallbacks.erase(handleId);
}

void VehiclePowertrainTransmissionPublisher::publishTorqueDistributionChanged(float TorqueDistribution) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueDistributionChanged(TorqueDistribution);
    }
    std::shared_lock<std::shared_timed_mutex> torqueDistributionCallbacksLock(m_torqueDistributionCallbacksMutex);
    const auto torqueDistributionCallbacks = m_torqueDistributionCallbacks;
    torqueDistributionCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueDistributionCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueDistribution);
        }
    }
}

