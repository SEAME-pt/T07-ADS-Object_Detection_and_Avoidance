

#include "vehicle/generated/core/vehiclemotionmanagementelectricaxlerow2.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleMotionManagementElectricAxleRow2Publisher::subscribeToAllChanges(IVehicleMotionManagementElectricAxleRow2Subscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleMotionManagementElectricAxleRow2Subscriber>(subscriber));
    }
}

void VehicleMotionManagementElectricAxleRow2Publisher::unsubscribeFromAllChanges(IVehicleMotionManagementElectricAxleRow2Subscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueChanged(VehicleMotionManagementElectricAxleRow2TorquePropertyCb callback)
{
    auto handleId = m_torqueChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueCallbacksMutex);
    m_torqueCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementElectricAxleRow2Publisher::unsubscribeFromTorqueChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueCallbacksMutex);
    m_torqueCallbacks.erase(handleId);
}

void VehicleMotionManagementElectricAxleRow2Publisher::publishTorqueChanged(int32_t Torque) const
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

long VehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueTargetChanged(VehicleMotionManagementElectricAxleRow2TorqueTargetPropertyCb callback)
{
    auto handleId = m_torqueTargetChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueTargetCallbacksMutex);
    m_torqueTargetCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementElectricAxleRow2Publisher::unsubscribeFromTorqueTargetChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueTargetCallbacksMutex);
    m_torqueTargetCallbacks.erase(handleId);
}

void VehicleMotionManagementElectricAxleRow2Publisher::publishTorqueTargetChanged(int32_t TorqueTarget) const
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

long VehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueMaximumChanged(VehicleMotionManagementElectricAxleRow2TorqueMaximumPropertyCb callback)
{
    auto handleId = m_torqueMaximumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMaximumCallbacksMutex);
    m_torqueMaximumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementElectricAxleRow2Publisher::unsubscribeFromTorqueMaximumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMaximumCallbacksMutex);
    m_torqueMaximumCallbacks.erase(handleId);
}

void VehicleMotionManagementElectricAxleRow2Publisher::publishTorqueMaximumChanged(int32_t TorqueMaximum) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueMaximumChanged(TorqueMaximum);
    }
    std::shared_lock<std::shared_timed_mutex> torqueMaximumCallbacksLock(m_torqueMaximumCallbacksMutex);
    const auto torqueMaximumCallbacks = m_torqueMaximumCallbacks;
    torqueMaximumCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueMaximumCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueMaximum);
        }
    }
}

long VehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueMinimumChanged(VehicleMotionManagementElectricAxleRow2TorqueMinimumPropertyCb callback)
{
    auto handleId = m_torqueMinimumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMinimumCallbacksMutex);
    m_torqueMinimumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementElectricAxleRow2Publisher::unsubscribeFromTorqueMinimumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMinimumCallbacksMutex);
    m_torqueMinimumCallbacks.erase(handleId);
}

void VehicleMotionManagementElectricAxleRow2Publisher::publishTorqueMinimumChanged(int32_t TorqueMinimum) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueMinimumChanged(TorqueMinimum);
    }
    std::shared_lock<std::shared_timed_mutex> torqueMinimumCallbacksLock(m_torqueMinimumCallbacksMutex);
    const auto torqueMinimumCallbacks = m_torqueMinimumCallbacks;
    torqueMinimumCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueMinimumCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueMinimum);
        }
    }
}

long VehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueLongTermMaximumChanged(VehicleMotionManagementElectricAxleRow2TorqueLongTermMaximumPropertyCb callback)
{
    auto handleId = m_torqueLongTermMaximumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueLongTermMaximumCallbacksMutex);
    m_torqueLongTermMaximumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementElectricAxleRow2Publisher::unsubscribeFromTorqueLongTermMaximumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueLongTermMaximumCallbacksMutex);
    m_torqueLongTermMaximumCallbacks.erase(handleId);
}

void VehicleMotionManagementElectricAxleRow2Publisher::publishTorqueLongTermMaximumChanged(int32_t TorqueLongTermMaximum) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueLongTermMaximumChanged(TorqueLongTermMaximum);
    }
    std::shared_lock<std::shared_timed_mutex> torqueLongTermMaximumCallbacksLock(m_torqueLongTermMaximumCallbacksMutex);
    const auto torqueLongTermMaximumCallbacks = m_torqueLongTermMaximumCallbacks;
    torqueLongTermMaximumCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueLongTermMaximumCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueLongTermMaximum);
        }
    }
}

long VehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueLongTermMinimumChanged(VehicleMotionManagementElectricAxleRow2TorqueLongTermMinimumPropertyCb callback)
{
    auto handleId = m_torqueLongTermMinimumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueLongTermMinimumCallbacksMutex);
    m_torqueLongTermMinimumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementElectricAxleRow2Publisher::unsubscribeFromTorqueLongTermMinimumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueLongTermMinimumCallbacksMutex);
    m_torqueLongTermMinimumCallbacks.erase(handleId);
}

void VehicleMotionManagementElectricAxleRow2Publisher::publishTorqueLongTermMinimumChanged(int32_t TorqueLongTermMinimum) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueLongTermMinimumChanged(TorqueLongTermMinimum);
    }
    std::shared_lock<std::shared_timed_mutex> torqueLongTermMinimumCallbacksLock(m_torqueLongTermMinimumCallbacksMutex);
    const auto torqueLongTermMinimumCallbacks = m_torqueLongTermMinimumCallbacks;
    torqueLongTermMinimumCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueLongTermMinimumCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueLongTermMinimum);
        }
    }
}

long VehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueShortTermMaximumChanged(VehicleMotionManagementElectricAxleRow2TorqueShortTermMaximumPropertyCb callback)
{
    auto handleId = m_torqueShortTermMaximumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueShortTermMaximumCallbacksMutex);
    m_torqueShortTermMaximumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementElectricAxleRow2Publisher::unsubscribeFromTorqueShortTermMaximumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueShortTermMaximumCallbacksMutex);
    m_torqueShortTermMaximumCallbacks.erase(handleId);
}

void VehicleMotionManagementElectricAxleRow2Publisher::publishTorqueShortTermMaximumChanged(int32_t TorqueShortTermMaximum) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueShortTermMaximumChanged(TorqueShortTermMaximum);
    }
    std::shared_lock<std::shared_timed_mutex> torqueShortTermMaximumCallbacksLock(m_torqueShortTermMaximumCallbacksMutex);
    const auto torqueShortTermMaximumCallbacks = m_torqueShortTermMaximumCallbacks;
    torqueShortTermMaximumCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueShortTermMaximumCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueShortTermMaximum);
        }
    }
}

long VehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueShortTermMinimumChanged(VehicleMotionManagementElectricAxleRow2TorqueShortTermMinimumPropertyCb callback)
{
    auto handleId = m_torqueShortTermMinimumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueShortTermMinimumCallbacksMutex);
    m_torqueShortTermMinimumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementElectricAxleRow2Publisher::unsubscribeFromTorqueShortTermMinimumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueShortTermMinimumCallbacksMutex);
    m_torqueShortTermMinimumCallbacks.erase(handleId);
}

void VehicleMotionManagementElectricAxleRow2Publisher::publishTorqueShortTermMinimumChanged(int32_t TorqueShortTermMinimum) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueShortTermMinimumChanged(TorqueShortTermMinimum);
    }
    std::shared_lock<std::shared_timed_mutex> torqueShortTermMinimumCallbacksLock(m_torqueShortTermMinimumCallbacksMutex);
    const auto torqueShortTermMinimumCallbacks = m_torqueShortTermMinimumCallbacks;
    torqueShortTermMinimumCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueShortTermMinimumCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueShortTermMinimum);
        }
    }
}

long VehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueMaximumLimitChanged(VehicleMotionManagementElectricAxleRow2TorqueMaximumLimitPropertyCb callback)
{
    auto handleId = m_torqueMaximumLimitChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMaximumLimitCallbacksMutex);
    m_torqueMaximumLimitCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementElectricAxleRow2Publisher::unsubscribeFromTorqueMaximumLimitChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMaximumLimitCallbacksMutex);
    m_torqueMaximumLimitCallbacks.erase(handleId);
}

void VehicleMotionManagementElectricAxleRow2Publisher::publishTorqueMaximumLimitChanged(int32_t TorqueMaximumLimit) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueMaximumLimitChanged(TorqueMaximumLimit);
    }
    std::shared_lock<std::shared_timed_mutex> torqueMaximumLimitCallbacksLock(m_torqueMaximumLimitCallbacksMutex);
    const auto torqueMaximumLimitCallbacks = m_torqueMaximumLimitCallbacks;
    torqueMaximumLimitCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueMaximumLimitCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueMaximumLimit);
        }
    }
}

long VehicleMotionManagementElectricAxleRow2Publisher::subscribeToTorqueMinimumLimitChanged(VehicleMotionManagementElectricAxleRow2TorqueMinimumLimitPropertyCb callback)
{
    auto handleId = m_torqueMinimumLimitChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMinimumLimitCallbacksMutex);
    m_torqueMinimumLimitCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementElectricAxleRow2Publisher::unsubscribeFromTorqueMinimumLimitChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueMinimumLimitCallbacksMutex);
    m_torqueMinimumLimitCallbacks.erase(handleId);
}

void VehicleMotionManagementElectricAxleRow2Publisher::publishTorqueMinimumLimitChanged(int32_t TorqueMinimumLimit) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueMinimumLimitChanged(TorqueMinimumLimit);
    }
    std::shared_lock<std::shared_timed_mutex> torqueMinimumLimitCallbacksLock(m_torqueMinimumLimitCallbacksMutex);
    const auto torqueMinimumLimitCallbacks = m_torqueMinimumLimitCallbacks;
    torqueMinimumLimitCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueMinimumLimitCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueMinimumLimit);
        }
    }
}

long VehicleMotionManagementElectricAxleRow2Publisher::subscribeToRotationalSpeedChanged(VehicleMotionManagementElectricAxleRow2RotationalSpeedPropertyCb callback)
{
    auto handleId = m_rotationalSpeedChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_rotationalSpeedCallbacksMutex);
    m_rotationalSpeedCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementElectricAxleRow2Publisher::unsubscribeFromRotationalSpeedChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rotationalSpeedCallbacksMutex);
    m_rotationalSpeedCallbacks.erase(handleId);
}

void VehicleMotionManagementElectricAxleRow2Publisher::publishRotationalSpeedChanged(int32_t RotationalSpeed) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onRotationalSpeedChanged(RotationalSpeed);
    }
    std::shared_lock<std::shared_timed_mutex> rotationalSpeedCallbacksLock(m_rotationalSpeedCallbacksMutex);
    const auto rotationalSpeedCallbacks = m_rotationalSpeedCallbacks;
    rotationalSpeedCallbacksLock.unlock();
    for(const auto& callbackEntry: rotationalSpeedCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(RotationalSpeed);
        }
    }
}

long VehicleMotionManagementElectricAxleRow2Publisher::subscribeToRotationalSpeedTargetChanged(VehicleMotionManagementElectricAxleRow2RotationalSpeedTargetPropertyCb callback)
{
    auto handleId = m_rotationalSpeedTargetChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_rotationalSpeedTargetCallbacksMutex);
    m_rotationalSpeedTargetCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementElectricAxleRow2Publisher::unsubscribeFromRotationalSpeedTargetChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rotationalSpeedTargetCallbacksMutex);
    m_rotationalSpeedTargetCallbacks.erase(handleId);
}

void VehicleMotionManagementElectricAxleRow2Publisher::publishRotationalSpeedTargetChanged(int32_t RotationalSpeedTarget) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onRotationalSpeedTargetChanged(RotationalSpeedTarget);
    }
    std::shared_lock<std::shared_timed_mutex> rotationalSpeedTargetCallbacksLock(m_rotationalSpeedTargetCallbacksMutex);
    const auto rotationalSpeedTargetCallbacks = m_rotationalSpeedTargetCallbacks;
    rotationalSpeedTargetCallbacksLock.unlock();
    for(const auto& callbackEntry: rotationalSpeedTargetCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(RotationalSpeedTarget);
        }
    }
}

long VehicleMotionManagementElectricAxleRow2Publisher::subscribeToRotationalSpeedMaximumLimitChanged(VehicleMotionManagementElectricAxleRow2RotationalSpeedMaximumLimitPropertyCb callback)
{
    auto handleId = m_rotationalSpeedMaximumLimitChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_rotationalSpeedMaximumLimitCallbacksMutex);
    m_rotationalSpeedMaximumLimitCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementElectricAxleRow2Publisher::unsubscribeFromRotationalSpeedMaximumLimitChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rotationalSpeedMaximumLimitCallbacksMutex);
    m_rotationalSpeedMaximumLimitCallbacks.erase(handleId);
}

void VehicleMotionManagementElectricAxleRow2Publisher::publishRotationalSpeedMaximumLimitChanged(int32_t RotationalSpeedMaximumLimit) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onRotationalSpeedMaximumLimitChanged(RotationalSpeedMaximumLimit);
    }
    std::shared_lock<std::shared_timed_mutex> rotationalSpeedMaximumLimitCallbacksLock(m_rotationalSpeedMaximumLimitCallbacksMutex);
    const auto rotationalSpeedMaximumLimitCallbacks = m_rotationalSpeedMaximumLimitCallbacks;
    rotationalSpeedMaximumLimitCallbacksLock.unlock();
    for(const auto& callbackEntry: rotationalSpeedMaximumLimitCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(RotationalSpeedMaximumLimit);
        }
    }
}

long VehicleMotionManagementElectricAxleRow2Publisher::subscribeToRotationalSpeedMinimumLimitChanged(VehicleMotionManagementElectricAxleRow2RotationalSpeedMinimumLimitPropertyCb callback)
{
    auto handleId = m_rotationalSpeedMinimumLimitChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_rotationalSpeedMinimumLimitCallbacksMutex);
    m_rotationalSpeedMinimumLimitCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementElectricAxleRow2Publisher::unsubscribeFromRotationalSpeedMinimumLimitChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_rotationalSpeedMinimumLimitCallbacksMutex);
    m_rotationalSpeedMinimumLimitCallbacks.erase(handleId);
}

void VehicleMotionManagementElectricAxleRow2Publisher::publishRotationalSpeedMinimumLimitChanged(int32_t RotationalSpeedMinimumLimit) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onRotationalSpeedMinimumLimitChanged(RotationalSpeedMinimumLimit);
    }
    std::shared_lock<std::shared_timed_mutex> rotationalSpeedMinimumLimitCallbacksLock(m_rotationalSpeedMinimumLimitCallbacksMutex);
    const auto rotationalSpeedMinimumLimitCallbacks = m_rotationalSpeedMinimumLimitCallbacks;
    rotationalSpeedMinimumLimitCallbacksLock.unlock();
    for(const auto& callbackEntry: rotationalSpeedMinimumLimitCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(RotationalSpeedMinimumLimit);
        }
    }
}

