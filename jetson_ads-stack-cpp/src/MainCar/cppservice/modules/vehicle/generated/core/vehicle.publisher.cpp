

#include "vehicle/generated/core/vehicle.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehiclePublisher::subscribeToAllChanges(IVehicleSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleSubscriber>(subscriber));
    }
}

void VehiclePublisher::unsubscribeFromAllChanges(IVehicleSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehiclePublisher::subscribeToLowVoltageSystemStateChanged(VehicleLowVoltageSystemStatePropertyCb callback)
{
    auto handleId = m_lowVoltageSystemStateChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_lowVoltageSystemStateCallbacksMutex);
    m_lowVoltageSystemStateCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePublisher::unsubscribeFromLowVoltageSystemStateChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_lowVoltageSystemStateCallbacksMutex);
    m_lowVoltageSystemStateCallbacks.erase(handleId);
}

void VehiclePublisher::publishLowVoltageSystemStateChanged(Vehicle_LowVoltageSystemState_ValueEnum LowVoltageSystemState) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onLowVoltageSystemStateChanged(LowVoltageSystemState);
    }
    std::shared_lock<std::shared_timed_mutex> lowVoltageSystemStateCallbacksLock(m_lowVoltageSystemStateCallbacksMutex);
    const auto lowVoltageSystemStateCallbacks = m_lowVoltageSystemStateCallbacks;
    lowVoltageSystemStateCallbacksLock.unlock();
    for(const auto& callbackEntry: lowVoltageSystemStateCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(LowVoltageSystemState);
        }
    }
}

long VehiclePublisher::subscribeToSpeedChanged(VehicleSpeedPropertyCb callback)
{
    auto handleId = m_speedChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_speedCallbacksMutex);
    m_speedCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePublisher::unsubscribeFromSpeedChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_speedCallbacksMutex);
    m_speedCallbacks.erase(handleId);
}

void VehiclePublisher::publishSpeedChanged(float Speed) const
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

long VehiclePublisher::subscribeToTraveledDistanceChanged(VehicleTraveledDistancePropertyCb callback)
{
    auto handleId = m_traveledDistanceChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_traveledDistanceCallbacksMutex);
    m_traveledDistanceCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePublisher::unsubscribeFromTraveledDistanceChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_traveledDistanceCallbacksMutex);
    m_traveledDistanceCallbacks.erase(handleId);
}

void VehiclePublisher::publishTraveledDistanceChanged(float TraveledDistance) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTraveledDistanceChanged(TraveledDistance);
    }
    std::shared_lock<std::shared_timed_mutex> traveledDistanceCallbacksLock(m_traveledDistanceCallbacksMutex);
    const auto traveledDistanceCallbacks = m_traveledDistanceCallbacks;
    traveledDistanceCallbacksLock.unlock();
    for(const auto& callbackEntry: traveledDistanceCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TraveledDistance);
        }
    }
}

long VehiclePublisher::subscribeToTraveledDistanceSinceStartChanged(VehicleTraveledDistanceSinceStartPropertyCb callback)
{
    auto handleId = m_traveledDistanceSinceStartChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_traveledDistanceSinceStartCallbacksMutex);
    m_traveledDistanceSinceStartCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePublisher::unsubscribeFromTraveledDistanceSinceStartChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_traveledDistanceSinceStartCallbacksMutex);
    m_traveledDistanceSinceStartCallbacks.erase(handleId);
}

void VehiclePublisher::publishTraveledDistanceSinceStartChanged(float TraveledDistanceSinceStart) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTraveledDistanceSinceStartChanged(TraveledDistanceSinceStart);
    }
    std::shared_lock<std::shared_timed_mutex> traveledDistanceSinceStartCallbacksLock(m_traveledDistanceSinceStartCallbacksMutex);
    const auto traveledDistanceSinceStartCallbacks = m_traveledDistanceSinceStartCallbacks;
    traveledDistanceSinceStartCallbacksLock.unlock();
    for(const auto& callbackEntry: traveledDistanceSinceStartCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TraveledDistanceSinceStart);
        }
    }
}

long VehiclePublisher::subscribeToStartTimeChanged(VehicleStartTimePropertyCb callback)
{
    auto handleId = m_startTimeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_startTimeCallbacksMutex);
    m_startTimeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePublisher::unsubscribeFromStartTimeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_startTimeCallbacksMutex);
    m_startTimeCallbacks.erase(handleId);
}

void VehiclePublisher::publishStartTimeChanged(const std::string& StartTime) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onStartTimeChanged(StartTime);
    }
    std::shared_lock<std::shared_timed_mutex> startTimeCallbacksLock(m_startTimeCallbacksMutex);
    const auto startTimeCallbacks = m_startTimeCallbacks;
    startTimeCallbacksLock.unlock();
    for(const auto& callbackEntry: startTimeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(StartTime);
        }
    }
}

long VehiclePublisher::subscribeToTripDurationChanged(VehicleTripDurationPropertyCb callback)
{
    auto handleId = m_tripDurationChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_tripDurationCallbacksMutex);
    m_tripDurationCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePublisher::unsubscribeFromTripDurationChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_tripDurationCallbacksMutex);
    m_tripDurationCallbacks.erase(handleId);
}

void VehiclePublisher::publishTripDurationChanged(float TripDuration) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTripDurationChanged(TripDuration);
    }
    std::shared_lock<std::shared_timed_mutex> tripDurationCallbacksLock(m_tripDurationCallbacksMutex);
    const auto tripDurationCallbacks = m_tripDurationCallbacks;
    tripDurationCallbacksLock.unlock();
    for(const auto& callbackEntry: tripDurationCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TripDuration);
        }
    }
}

long VehiclePublisher::subscribeToTripMeterReadingChanged(VehicleTripMeterReadingPropertyCb callback)
{
    auto handleId = m_tripMeterReadingChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_tripMeterReadingCallbacksMutex);
    m_tripMeterReadingCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePublisher::unsubscribeFromTripMeterReadingChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_tripMeterReadingCallbacksMutex);
    m_tripMeterReadingCallbacks.erase(handleId);
}

void VehiclePublisher::publishTripMeterReadingChanged(float TripMeterReading) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTripMeterReadingChanged(TripMeterReading);
    }
    std::shared_lock<std::shared_timed_mutex> tripMeterReadingCallbacksLock(m_tripMeterReadingCallbacksMutex);
    const auto tripMeterReadingCallbacks = m_tripMeterReadingCallbacks;
    tripMeterReadingCallbacksLock.unlock();
    for(const auto& callbackEntry: tripMeterReadingCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TripMeterReading);
        }
    }
}

long VehiclePublisher::subscribeToIsBrokenDownChanged(VehicleIsBrokenDownPropertyCb callback)
{
    auto handleId = m_isBrokenDownChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isBrokenDownCallbacksMutex);
    m_isBrokenDownCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePublisher::unsubscribeFromIsBrokenDownChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isBrokenDownCallbacksMutex);
    m_isBrokenDownCallbacks.erase(handleId);
}

void VehiclePublisher::publishIsBrokenDownChanged(bool IsBrokenDown) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsBrokenDownChanged(IsBrokenDown);
    }
    std::shared_lock<std::shared_timed_mutex> isBrokenDownCallbacksLock(m_isBrokenDownCallbacksMutex);
    const auto isBrokenDownCallbacks = m_isBrokenDownCallbacks;
    isBrokenDownCallbacksLock.unlock();
    for(const auto& callbackEntry: isBrokenDownCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsBrokenDown);
        }
    }
}

long VehiclePublisher::subscribeToIsMovingChanged(VehicleIsMovingPropertyCb callback)
{
    auto handleId = m_isMovingChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isMovingCallbacksMutex);
    m_isMovingCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePublisher::unsubscribeFromIsMovingChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isMovingCallbacksMutex);
    m_isMovingCallbacks.erase(handleId);
}

void VehiclePublisher::publishIsMovingChanged(bool IsMoving) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIsMovingChanged(IsMoving);
    }
    std::shared_lock<std::shared_timed_mutex> isMovingCallbacksLock(m_isMovingCallbacksMutex);
    const auto isMovingCallbacks = m_isMovingCallbacks;
    isMovingCallbacksLock.unlock();
    for(const auto& callbackEntry: isMovingCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(IsMoving);
        }
    }
}

long VehiclePublisher::subscribeToAverageSpeedChanged(VehicleAverageSpeedPropertyCb callback)
{
    auto handleId = m_averageSpeedChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_averageSpeedCallbacksMutex);
    m_averageSpeedCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePublisher::unsubscribeFromAverageSpeedChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_averageSpeedCallbacksMutex);
    m_averageSpeedCallbacks.erase(handleId);
}

void VehiclePublisher::publishAverageSpeedChanged(float AverageSpeed) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onAverageSpeedChanged(AverageSpeed);
    }
    std::shared_lock<std::shared_timed_mutex> averageSpeedCallbacksLock(m_averageSpeedCallbacksMutex);
    const auto averageSpeedCallbacks = m_averageSpeedCallbacks;
    averageSpeedCallbacksLock.unlock();
    for(const auto& callbackEntry: averageSpeedCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(AverageSpeed);
        }
    }
}

long VehiclePublisher::subscribeToPowerOptimizeLevelChanged(VehiclePowerOptimizeLevelPropertyCb callback)
{
    auto handleId = m_powerOptimizeLevelChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_powerOptimizeLevelCallbacksMutex);
    m_powerOptimizeLevelCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePublisher::unsubscribeFromPowerOptimizeLevelChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_powerOptimizeLevelCallbacksMutex);
    m_powerOptimizeLevelCallbacks.erase(handleId);
}

void VehiclePublisher::publishPowerOptimizeLevelChanged(int32_t PowerOptimizeLevel) const
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

long VehiclePublisher::subscribeToIsAutoPowerOptimizeChanged(VehicleIsAutoPowerOptimizePropertyCb callback)
{
    auto handleId = m_isAutoPowerOptimizeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_isAutoPowerOptimizeCallbacksMutex);
    m_isAutoPowerOptimizeCallbacks[handleId] = callback;
    return handleId;
}

void VehiclePublisher::unsubscribeFromIsAutoPowerOptimizeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_isAutoPowerOptimizeCallbacksMutex);
    m_isAutoPowerOptimizeCallbacks.erase(handleId);
}

void VehiclePublisher::publishIsAutoPowerOptimizeChanged(bool IsAutoPowerOptimize) const
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

