

#include "vehicle/generated/core/vehiclemotionmanagementsteeringsteeringwheel.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToAllChanges(IVehicleMotionManagementSteeringSteeringWheelSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleMotionManagementSteeringSteeringWheelSubscriber>(subscriber));
    }
}

void VehicleMotionManagementSteeringSteeringWheelPublisher::unsubscribeFromAllChanges(IVehicleMotionManagementSteeringSteeringWheelSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToAngleChanged(VehicleMotionManagementSteeringSteeringWheelAnglePropertyCb callback)
{
    auto handleId = m_angleChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_angleCallbacksMutex);
    m_angleCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementSteeringSteeringWheelPublisher::unsubscribeFromAngleChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_angleCallbacksMutex);
    m_angleCallbacks.erase(handleId);
}

void VehicleMotionManagementSteeringSteeringWheelPublisher::publishAngleChanged(int32_t Angle) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onAngleChanged(Angle);
    }
    std::shared_lock<std::shared_timed_mutex> angleCallbacksLock(m_angleCallbacksMutex);
    const auto angleCallbacks = m_angleCallbacks;
    angleCallbacksLock.unlock();
    for(const auto& callbackEntry: angleCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Angle);
        }
    }
}

long VehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToAngleTargetChanged(VehicleMotionManagementSteeringSteeringWheelAngleTargetPropertyCb callback)
{
    auto handleId = m_angleTargetChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_angleTargetCallbacksMutex);
    m_angleTargetCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementSteeringSteeringWheelPublisher::unsubscribeFromAngleTargetChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_angleTargetCallbacksMutex);
    m_angleTargetCallbacks.erase(handleId);
}

void VehicleMotionManagementSteeringSteeringWheelPublisher::publishAngleTargetChanged(int32_t AngleTarget) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onAngleTargetChanged(AngleTarget);
    }
    std::shared_lock<std::shared_timed_mutex> angleTargetCallbacksLock(m_angleTargetCallbacksMutex);
    const auto angleTargetCallbacks = m_angleTargetCallbacks;
    angleTargetCallbacksLock.unlock();
    for(const auto& callbackEntry: angleTargetCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(AngleTarget);
        }
    }
}

long VehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToAngleTargetModeChanged(VehicleMotionManagementSteeringSteeringWheelAngleTargetModePropertyCb callback)
{
    auto handleId = m_angleTargetModeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_angleTargetModeCallbacksMutex);
    m_angleTargetModeCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementSteeringSteeringWheelPublisher::unsubscribeFromAngleTargetModeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_angleTargetModeCallbacksMutex);
    m_angleTargetModeCallbacks.erase(handleId);
}

void VehicleMotionManagementSteeringSteeringWheelPublisher::publishAngleTargetModeChanged(int32_t AngleTargetMode) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onAngleTargetModeChanged(AngleTargetMode);
    }
    std::shared_lock<std::shared_timed_mutex> angleTargetModeCallbacksLock(m_angleTargetModeCallbacksMutex);
    const auto angleTargetModeCallbacks = m_angleTargetModeCallbacks;
    angleTargetModeCallbacksLock.unlock();
    for(const auto& callbackEntry: angleTargetModeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(AngleTargetMode);
        }
    }
}

long VehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToTorqueOffsetTargetChanged(VehicleMotionManagementSteeringSteeringWheelTorqueOffsetTargetPropertyCb callback)
{
    auto handleId = m_torqueOffsetTargetChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueOffsetTargetCallbacksMutex);
    m_torqueOffsetTargetCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementSteeringSteeringWheelPublisher::unsubscribeFromTorqueOffsetTargetChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueOffsetTargetCallbacksMutex);
    m_torqueOffsetTargetCallbacks.erase(handleId);
}

void VehicleMotionManagementSteeringSteeringWheelPublisher::publishTorqueOffsetTargetChanged(int32_t TorqueOffsetTarget) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueOffsetTargetChanged(TorqueOffsetTarget);
    }
    std::shared_lock<std::shared_timed_mutex> torqueOffsetTargetCallbacksLock(m_torqueOffsetTargetCallbacksMutex);
    const auto torqueOffsetTargetCallbacks = m_torqueOffsetTargetCallbacks;
    torqueOffsetTargetCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueOffsetTargetCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueOffsetTarget);
        }
    }
}

long VehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToTorqueOffsetTargetModeChanged(VehicleMotionManagementSteeringSteeringWheelTorqueOffsetTargetModePropertyCb callback)
{
    auto handleId = m_torqueOffsetTargetModeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueOffsetTargetModeCallbacksMutex);
    m_torqueOffsetTargetModeCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementSteeringSteeringWheelPublisher::unsubscribeFromTorqueOffsetTargetModeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueOffsetTargetModeCallbacksMutex);
    m_torqueOffsetTargetModeCallbacks.erase(handleId);
}

void VehicleMotionManagementSteeringSteeringWheelPublisher::publishTorqueOffsetTargetModeChanged(int32_t TorqueOffsetTargetMode) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueOffsetTargetModeChanged(TorqueOffsetTargetMode);
    }
    std::shared_lock<std::shared_timed_mutex> torqueOffsetTargetModeCallbacksLock(m_torqueOffsetTargetModeCallbacksMutex);
    const auto torqueOffsetTargetModeCallbacks = m_torqueOffsetTargetModeCallbacks;
    torqueOffsetTargetModeCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueOffsetTargetModeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueOffsetTargetMode);
        }
    }
}

long VehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToTorqueTargetChanged(VehicleMotionManagementSteeringSteeringWheelTorqueTargetPropertyCb callback)
{
    auto handleId = m_torqueTargetChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueTargetCallbacksMutex);
    m_torqueTargetCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementSteeringSteeringWheelPublisher::unsubscribeFromTorqueTargetChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueTargetCallbacksMutex);
    m_torqueTargetCallbacks.erase(handleId);
}

void VehicleMotionManagementSteeringSteeringWheelPublisher::publishTorqueTargetChanged(int32_t TorqueTarget) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueTargetChanged(TorqueTarget);
    }
    std::shared_lock<std::shared_timed_mutex> torqueTargetCallbacksLock(m_torqueTargetCallbacksMutex);
    const auto torqueTargetCallbacks = m_torqueTargetCallbacks;
    torqueTargetCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueTargetCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueTarget);
        }
    }
}

long VehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToTorqueTargetModeChanged(VehicleMotionManagementSteeringSteeringWheelTorqueTargetModePropertyCb callback)
{
    auto handleId = m_torqueTargetModeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueTargetModeCallbacksMutex);
    m_torqueTargetModeCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementSteeringSteeringWheelPublisher::unsubscribeFromTorqueTargetModeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueTargetModeCallbacksMutex);
    m_torqueTargetModeCallbacks.erase(handleId);
}

void VehicleMotionManagementSteeringSteeringWheelPublisher::publishTorqueTargetModeChanged(int32_t TorqueTargetMode) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueTargetModeChanged(TorqueTargetMode);
    }
    std::shared_lock<std::shared_timed_mutex> torqueTargetModeCallbacksLock(m_torqueTargetModeCallbacksMutex);
    const auto torqueTargetModeCallbacks = m_torqueTargetModeCallbacks;
    torqueTargetModeCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueTargetModeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueTargetMode);
        }
    }
}

long VehicleMotionManagementSteeringSteeringWheelPublisher::subscribeToTorqueChanged(VehicleMotionManagementSteeringSteeringWheelTorquePropertyCb callback)
{
    auto handleId = m_torqueChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueCallbacksMutex);
    m_torqueCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementSteeringSteeringWheelPublisher::unsubscribeFromTorqueChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueCallbacksMutex);
    m_torqueCallbacks.erase(handleId);
}

void VehicleMotionManagementSteeringSteeringWheelPublisher::publishTorqueChanged(int32_t Torque) const
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

