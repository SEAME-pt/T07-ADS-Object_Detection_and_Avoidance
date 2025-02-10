

#include "vehicle/generated/core/vehiclemotionmanagementsteeringaxlerow1.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleMotionManagementSteeringAxleRow1Publisher::subscribeToAllChanges(IVehicleMotionManagementSteeringAxleRow1Subscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleMotionManagementSteeringAxleRow1Subscriber>(subscriber));
    }
}

void VehicleMotionManagementSteeringAxleRow1Publisher::unsubscribeFromAllChanges(IVehicleMotionManagementSteeringAxleRow1Subscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleMotionManagementSteeringAxleRow1Publisher::subscribeToRackPositionOffsetTargetChanged(VehicleMotionManagementSteeringAxleRow1RackPositionOffsetTargetPropertyCb callback)
{
    auto handleId = m_rackPositionOffsetTargetChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_rackPositionOffsetTargetCallbacksMutex);
    m_rackPositionOffsetTargetCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementSteeringAxleRow1Publisher::unsubscribeFromRackPositionOffsetTargetChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rackPositionOffsetTargetCallbacksMutex);
    m_rackPositionOffsetTargetCallbacks.erase(handleId);
}

void VehicleMotionManagementSteeringAxleRow1Publisher::publishRackPositionOffsetTargetChanged(int32_t RackPositionOffsetTarget) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onRackPositionOffsetTargetChanged(RackPositionOffsetTarget);
    }
    std::shared_lock<std::shared_timed_mutex> rackPositionOffsetTargetCallbacksLock(m_rackPositionOffsetTargetCallbacksMutex);
    const auto rackPositionOffsetTargetCallbacks = m_rackPositionOffsetTargetCallbacks;
    rackPositionOffsetTargetCallbacksLock.unlock();
    for(const auto& callbackEntry: rackPositionOffsetTargetCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(RackPositionOffsetTarget);
        }
    }
}

long VehicleMotionManagementSteeringAxleRow1Publisher::subscribeToRackPositionOffsetTargetModeChanged(VehicleMotionManagementSteeringAxleRow1RackPositionOffsetTargetModePropertyCb callback)
{
    auto handleId = m_rackPositionOffsetTargetModeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_rackPositionOffsetTargetModeCallbacksMutex);
    m_rackPositionOffsetTargetModeCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementSteeringAxleRow1Publisher::unsubscribeFromRackPositionOffsetTargetModeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rackPositionOffsetTargetModeCallbacksMutex);
    m_rackPositionOffsetTargetModeCallbacks.erase(handleId);
}

void VehicleMotionManagementSteeringAxleRow1Publisher::publishRackPositionOffsetTargetModeChanged(int32_t RackPositionOffsetTargetMode) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onRackPositionOffsetTargetModeChanged(RackPositionOffsetTargetMode);
    }
    std::shared_lock<std::shared_timed_mutex> rackPositionOffsetTargetModeCallbacksLock(m_rackPositionOffsetTargetModeCallbacksMutex);
    const auto rackPositionOffsetTargetModeCallbacks = m_rackPositionOffsetTargetModeCallbacks;
    rackPositionOffsetTargetModeCallbacksLock.unlock();
    for(const auto& callbackEntry: rackPositionOffsetTargetModeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(RackPositionOffsetTargetMode);
        }
    }
}

long VehicleMotionManagementSteeringAxleRow1Publisher::subscribeToRackPositionChanged(VehicleMotionManagementSteeringAxleRow1RackPositionPropertyCb callback)
{
    auto handleId = m_rackPositionChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_rackPositionCallbacksMutex);
    m_rackPositionCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementSteeringAxleRow1Publisher::unsubscribeFromRackPositionChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rackPositionCallbacksMutex);
    m_rackPositionCallbacks.erase(handleId);
}

void VehicleMotionManagementSteeringAxleRow1Publisher::publishRackPositionChanged(int32_t RackPosition) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onRackPositionChanged(RackPosition);
    }
    std::shared_lock<std::shared_timed_mutex> rackPositionCallbacksLock(m_rackPositionCallbacksMutex);
    const auto rackPositionCallbacks = m_rackPositionCallbacks;
    rackPositionCallbacksLock.unlock();
    for(const auto& callbackEntry: rackPositionCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(RackPosition);
        }
    }
}

long VehicleMotionManagementSteeringAxleRow1Publisher::subscribeToRackPositionTargetChanged(VehicleMotionManagementSteeringAxleRow1RackPositionTargetPropertyCb callback)
{
    auto handleId = m_rackPositionTargetChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_rackPositionTargetCallbacksMutex);
    m_rackPositionTargetCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementSteeringAxleRow1Publisher::unsubscribeFromRackPositionTargetChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rackPositionTargetCallbacksMutex);
    m_rackPositionTargetCallbacks.erase(handleId);
}

void VehicleMotionManagementSteeringAxleRow1Publisher::publishRackPositionTargetChanged(int32_t RackPositionTarget) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onRackPositionTargetChanged(RackPositionTarget);
    }
    std::shared_lock<std::shared_timed_mutex> rackPositionTargetCallbacksLock(m_rackPositionTargetCallbacksMutex);
    const auto rackPositionTargetCallbacks = m_rackPositionTargetCallbacks;
    rackPositionTargetCallbacksLock.unlock();
    for(const auto& callbackEntry: rackPositionTargetCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(RackPositionTarget);
        }
    }
}

long VehicleMotionManagementSteeringAxleRow1Publisher::subscribeToRackPositionTargetModeChanged(VehicleMotionManagementSteeringAxleRow1RackPositionTargetModePropertyCb callback)
{
    auto handleId = m_rackPositionTargetModeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_rackPositionTargetModeCallbacksMutex);
    m_rackPositionTargetModeCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementSteeringAxleRow1Publisher::unsubscribeFromRackPositionTargetModeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rackPositionTargetModeCallbacksMutex);
    m_rackPositionTargetModeCallbacks.erase(handleId);
}

void VehicleMotionManagementSteeringAxleRow1Publisher::publishRackPositionTargetModeChanged(int32_t RackPositionTargetMode) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onRackPositionTargetModeChanged(RackPositionTargetMode);
    }
    std::shared_lock<std::shared_timed_mutex> rackPositionTargetModeCallbacksLock(m_rackPositionTargetModeCallbacksMutex);
    const auto rackPositionTargetModeCallbacks = m_rackPositionTargetModeCallbacks;
    rackPositionTargetModeCallbacksLock.unlock();
    for(const auto& callbackEntry: rackPositionTargetModeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(RackPositionTargetMode);
        }
    }
}

