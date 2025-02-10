

#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow1.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleMotionManagementBrakeAxleRow1Publisher::subscribeToAllChanges(IVehicleMotionManagementBrakeAxleRow1Subscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleMotionManagementBrakeAxleRow1Subscriber>(subscriber));
    }
}

void VehicleMotionManagementBrakeAxleRow1Publisher::unsubscribeFromAllChanges(IVehicleMotionManagementBrakeAxleRow1Subscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleMotionManagementBrakeAxleRow1Publisher::subscribeToTorqueElectricMinimumChanged(VehicleMotionManagementBrakeAxleRow1TorqueElectricMinimumPropertyCb callback)
{
    auto handleId = m_torqueElectricMinimumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueElectricMinimumCallbacksMutex);
    m_torqueElectricMinimumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakeAxleRow1Publisher::unsubscribeFromTorqueElectricMinimumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueElectricMinimumCallbacksMutex);
    m_torqueElectricMinimumCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakeAxleRow1Publisher::publishTorqueElectricMinimumChanged(int32_t TorqueElectricMinimum) const
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

long VehicleMotionManagementBrakeAxleRow1Publisher::subscribeToTorqueFrictionDifferenceMaximumChanged(VehicleMotionManagementBrakeAxleRow1TorqueFrictionDifferenceMaximumPropertyCb callback)
{
    auto handleId = m_torqueFrictionDifferenceMaximumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionDifferenceMaximumCallbacksMutex);
    m_torqueFrictionDifferenceMaximumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakeAxleRow1Publisher::unsubscribeFromTorqueFrictionDifferenceMaximumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionDifferenceMaximumCallbacksMutex);
    m_torqueFrictionDifferenceMaximumCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakeAxleRow1Publisher::publishTorqueFrictionDifferenceMaximumChanged(int32_t TorqueFrictionDifferenceMaximum) const
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

long VehicleMotionManagementBrakeAxleRow1Publisher::subscribeToTorqueDistributionFrictionRightMaximumChanged(VehicleMotionManagementBrakeAxleRow1TorqueDistributionFrictionRightMaximumPropertyCb callback)
{
    auto handleId = m_torqueDistributionFrictionRightMaximumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueDistributionFrictionRightMaximumCallbacksMutex);
    m_torqueDistributionFrictionRightMaximumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakeAxleRow1Publisher::unsubscribeFromTorqueDistributionFrictionRightMaximumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueDistributionFrictionRightMaximumCallbacksMutex);
    m_torqueDistributionFrictionRightMaximumCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakeAxleRow1Publisher::publishTorqueDistributionFrictionRightMaximumChanged(int32_t TorqueDistributionFrictionRightMaximum) const
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

long VehicleMotionManagementBrakeAxleRow1Publisher::subscribeToTorqueDistributionFrictionRightMinimumChanged(VehicleMotionManagementBrakeAxleRow1TorqueDistributionFrictionRightMinimumPropertyCb callback)
{
    auto handleId = m_torqueDistributionFrictionRightMinimumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueDistributionFrictionRightMinimumCallbacksMutex);
    m_torqueDistributionFrictionRightMinimumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakeAxleRow1Publisher::unsubscribeFromTorqueDistributionFrictionRightMinimumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueDistributionFrictionRightMinimumCallbacksMutex);
    m_torqueDistributionFrictionRightMinimumCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakeAxleRow1Publisher::publishTorqueDistributionFrictionRightMinimumChanged(int32_t TorqueDistributionFrictionRightMinimum) const
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

