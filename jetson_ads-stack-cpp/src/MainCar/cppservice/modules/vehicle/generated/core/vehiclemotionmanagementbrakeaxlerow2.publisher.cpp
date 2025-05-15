

#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow2.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleMotionManagementBrakeAxleRow2Publisher::subscribeToAllChanges(IVehicleMotionManagementBrakeAxleRow2Subscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleMotionManagementBrakeAxleRow2Subscriber>(subscriber));
    }
}

void VehicleMotionManagementBrakeAxleRow2Publisher::unsubscribeFromAllChanges(IVehicleMotionManagementBrakeAxleRow2Subscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleMotionManagementBrakeAxleRow2Publisher::subscribeToTorqueElectricMinimumChanged(VehicleMotionManagementBrakeAxleRow2TorqueElectricMinimumPropertyCb callback)
{
    auto handleId = m_torqueElectricMinimumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueElectricMinimumCallbacksMutex);
    m_torqueElectricMinimumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakeAxleRow2Publisher::unsubscribeFromTorqueElectricMinimumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueElectricMinimumCallbacksMutex);
    m_torqueElectricMinimumCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakeAxleRow2Publisher::publishTorqueElectricMinimumChanged(int32_t TorqueElectricMinimum) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueElectricMinimumChanged(TorqueElectricMinimum);
    }
    std::shared_lock<std::shared_timed_mutex> torqueElectricMinimumCallbacksLock(m_torqueElectricMinimumCallbacksMutex);
    const auto torqueElectricMinimumCallbacks = m_torqueElectricMinimumCallbacks;
    torqueElectricMinimumCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueElectricMinimumCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueElectricMinimum);
        }
    }
}

long VehicleMotionManagementBrakeAxleRow2Publisher::subscribeToTorqueFrictionDifferenceMaximumChanged(VehicleMotionManagementBrakeAxleRow2TorqueFrictionDifferenceMaximumPropertyCb callback)
{
    auto handleId = m_torqueFrictionDifferenceMaximumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionDifferenceMaximumCallbacksMutex);
    m_torqueFrictionDifferenceMaximumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakeAxleRow2Publisher::unsubscribeFromTorqueFrictionDifferenceMaximumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionDifferenceMaximumCallbacksMutex);
    m_torqueFrictionDifferenceMaximumCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakeAxleRow2Publisher::publishTorqueFrictionDifferenceMaximumChanged(int32_t TorqueFrictionDifferenceMaximum) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueFrictionDifferenceMaximumChanged(TorqueFrictionDifferenceMaximum);
    }
    std::shared_lock<std::shared_timed_mutex> torqueFrictionDifferenceMaximumCallbacksLock(m_torqueFrictionDifferenceMaximumCallbacksMutex);
    const auto torqueFrictionDifferenceMaximumCallbacks = m_torqueFrictionDifferenceMaximumCallbacks;
    torqueFrictionDifferenceMaximumCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueFrictionDifferenceMaximumCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueFrictionDifferenceMaximum);
        }
    }
}

long VehicleMotionManagementBrakeAxleRow2Publisher::subscribeToTorqueDistributionFrictionRightMaximumChanged(VehicleMotionManagementBrakeAxleRow2TorqueDistributionFrictionRightMaximumPropertyCb callback)
{
    auto handleId = m_torqueDistributionFrictionRightMaximumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueDistributionFrictionRightMaximumCallbacksMutex);
    m_torqueDistributionFrictionRightMaximumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakeAxleRow2Publisher::unsubscribeFromTorqueDistributionFrictionRightMaximumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueDistributionFrictionRightMaximumCallbacksMutex);
    m_torqueDistributionFrictionRightMaximumCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakeAxleRow2Publisher::publishTorqueDistributionFrictionRightMaximumChanged(int32_t TorqueDistributionFrictionRightMaximum) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueDistributionFrictionRightMaximumChanged(TorqueDistributionFrictionRightMaximum);
    }
    std::shared_lock<std::shared_timed_mutex> torqueDistributionFrictionRightMaximumCallbacksLock(m_torqueDistributionFrictionRightMaximumCallbacksMutex);
    const auto torqueDistributionFrictionRightMaximumCallbacks = m_torqueDistributionFrictionRightMaximumCallbacks;
    torqueDistributionFrictionRightMaximumCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueDistributionFrictionRightMaximumCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueDistributionFrictionRightMaximum);
        }
    }
}

long VehicleMotionManagementBrakeAxleRow2Publisher::subscribeToTorqueDistributionFrictionRightMinimumChanged(VehicleMotionManagementBrakeAxleRow2TorqueDistributionFrictionRightMinimumPropertyCb callback)
{
    auto handleId = m_torqueDistributionFrictionRightMinimumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueDistributionFrictionRightMinimumCallbacksMutex);
    m_torqueDistributionFrictionRightMinimumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakeAxleRow2Publisher::unsubscribeFromTorqueDistributionFrictionRightMinimumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueDistributionFrictionRightMinimumCallbacksMutex);
    m_torqueDistributionFrictionRightMinimumCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakeAxleRow2Publisher::publishTorqueDistributionFrictionRightMinimumChanged(int32_t TorqueDistributionFrictionRightMinimum) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueDistributionFrictionRightMinimumChanged(TorqueDistributionFrictionRightMinimum);
    }
    std::shared_lock<std::shared_timed_mutex> torqueDistributionFrictionRightMinimumCallbacksLock(m_torqueDistributionFrictionRightMinimumCallbacksMutex);
    const auto torqueDistributionFrictionRightMinimumCallbacks = m_torqueDistributionFrictionRightMinimumCallbacks;
    torqueDistributionFrictionRightMinimumCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueDistributionFrictionRightMinimumCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueDistributionFrictionRightMinimum);
        }
    }
}

