

#include "vehicle/generated/core/vehiclemotionmanagementbrake.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleMotionManagementBrakePublisher::subscribeToAllChanges(IVehicleMotionManagementBrakeSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleMotionManagementBrakeSubscriber>(subscriber));
    }
}

void VehicleMotionManagementBrakePublisher::unsubscribeFromAllChanges(IVehicleMotionManagementBrakeSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleMotionManagementBrakePublisher::subscribeToVehicleForceMaximumChanged(VehicleMotionManagementBrakeVehicleForceMaximumPropertyCb callback)
{
    auto handleId = m_vehicleForceMaximumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleForceMaximumCallbacksMutex);
    m_vehicleForceMaximumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakePublisher::unsubscribeFromVehicleForceMaximumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleForceMaximumCallbacksMutex);
    m_vehicleForceMaximumCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakePublisher::publishVehicleForceMaximumChanged(int32_t VehicleForceMaximum) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onVehicleForceMaximumChanged(VehicleForceMaximum);
    }
    std::shared_lock<std::shared_timed_mutex> vehicleForceMaximumCallbacksLock(m_vehicleForceMaximumCallbacksMutex);
    const auto vehicleForceMaximumCallbacks = m_vehicleForceMaximumCallbacks;
    vehicleForceMaximumCallbacksLock.unlock();
    for(const auto& callbackEntry: vehicleForceMaximumCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(VehicleForceMaximum);
        }
    }
}

long VehicleMotionManagementBrakePublisher::subscribeToVehicleForceElectricMinimumArbitratedChanged(VehicleMotionManagementBrakeVehicleForceElectricMinimumArbitratedPropertyCb callback)
{
    auto handleId = m_vehicleForceElectricMinimumArbitratedChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleForceElectricMinimumArbitratedCallbacksMutex);
    m_vehicleForceElectricMinimumArbitratedCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakePublisher::unsubscribeFromVehicleForceElectricMinimumArbitratedChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleForceElectricMinimumArbitratedCallbacksMutex);
    m_vehicleForceElectricMinimumArbitratedCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakePublisher::publishVehicleForceElectricMinimumArbitratedChanged(int32_t VehicleForceElectricMinimumArbitrated) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onVehicleForceElectricMinimumArbitratedChanged(VehicleForceElectricMinimumArbitrated);
    }
    std::shared_lock<std::shared_timed_mutex> vehicleForceElectricMinimumArbitratedCallbacksLock(m_vehicleForceElectricMinimumArbitratedCallbacksMutex);
    const auto vehicleForceElectricMinimumArbitratedCallbacks = m_vehicleForceElectricMinimumArbitratedCallbacks;
    vehicleForceElectricMinimumArbitratedCallbacksLock.unlock();
    for(const auto& callbackEntry: vehicleForceElectricMinimumArbitratedCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(VehicleForceElectricMinimumArbitrated);
        }
    }
}

long VehicleMotionManagementBrakePublisher::subscribeToVehicleForceElectricChanged(VehicleMotionManagementBrakeVehicleForceElectricPropertyCb callback)
{
    auto handleId = m_vehicleForceElectricChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleForceElectricCallbacksMutex);
    m_vehicleForceElectricCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakePublisher::unsubscribeFromVehicleForceElectricChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleForceElectricCallbacksMutex);
    m_vehicleForceElectricCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakePublisher::publishVehicleForceElectricChanged(int32_t VehicleForceElectric) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onVehicleForceElectricChanged(VehicleForceElectric);
    }
    std::shared_lock<std::shared_timed_mutex> vehicleForceElectricCallbacksLock(m_vehicleForceElectricCallbacksMutex);
    const auto vehicleForceElectricCallbacks = m_vehicleForceElectricCallbacks;
    vehicleForceElectricCallbacksLock.unlock();
    for(const auto& callbackEntry: vehicleForceElectricCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(VehicleForceElectric);
        }
    }
}

long VehicleMotionManagementBrakePublisher::subscribeToVehicleForceDistributionFrontMaximumChanged(VehicleMotionManagementBrakeVehicleForceDistributionFrontMaximumPropertyCb callback)
{
    auto handleId = m_vehicleForceDistributionFrontMaximumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleForceDistributionFrontMaximumCallbacksMutex);
    m_vehicleForceDistributionFrontMaximumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakePublisher::unsubscribeFromVehicleForceDistributionFrontMaximumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleForceDistributionFrontMaximumCallbacksMutex);
    m_vehicleForceDistributionFrontMaximumCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakePublisher::publishVehicleForceDistributionFrontMaximumChanged(int32_t VehicleForceDistributionFrontMaximum) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onVehicleForceDistributionFrontMaximumChanged(VehicleForceDistributionFrontMaximum);
    }
    std::shared_lock<std::shared_timed_mutex> vehicleForceDistributionFrontMaximumCallbacksLock(m_vehicleForceDistributionFrontMaximumCallbacksMutex);
    const auto vehicleForceDistributionFrontMaximumCallbacks = m_vehicleForceDistributionFrontMaximumCallbacks;
    vehicleForceDistributionFrontMaximumCallbacksLock.unlock();
    for(const auto& callbackEntry: vehicleForceDistributionFrontMaximumCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(VehicleForceDistributionFrontMaximum);
        }
    }
}

long VehicleMotionManagementBrakePublisher::subscribeToVehicleForceDistributionFrontMinimumChanged(VehicleMotionManagementBrakeVehicleForceDistributionFrontMinimumPropertyCb callback)
{
    auto handleId = m_vehicleForceDistributionFrontMinimumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleForceDistributionFrontMinimumCallbacksMutex);
    m_vehicleForceDistributionFrontMinimumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakePublisher::unsubscribeFromVehicleForceDistributionFrontMinimumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleForceDistributionFrontMinimumCallbacksMutex);
    m_vehicleForceDistributionFrontMinimumCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakePublisher::publishVehicleForceDistributionFrontMinimumChanged(int32_t VehicleForceDistributionFrontMinimum) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onVehicleForceDistributionFrontMinimumChanged(VehicleForceDistributionFrontMinimum);
    }
    std::shared_lock<std::shared_timed_mutex> vehicleForceDistributionFrontMinimumCallbacksLock(m_vehicleForceDistributionFrontMinimumCallbacksMutex);
    const auto vehicleForceDistributionFrontMinimumCallbacks = m_vehicleForceDistributionFrontMinimumCallbacks;
    vehicleForceDistributionFrontMinimumCallbacksLock.unlock();
    for(const auto& callbackEntry: vehicleForceDistributionFrontMinimumCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(VehicleForceDistributionFrontMinimum);
        }
    }
}

