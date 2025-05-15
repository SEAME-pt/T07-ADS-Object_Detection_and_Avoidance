

#include "vehicle/generated/core/vehiclemotionmanagementbrakeaxlerow1wheelleft.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::subscribeToAllChanges(IVehicleMotionManagementBrakeAxleRow1WheelLeftSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleMotionManagementBrakeAxleRow1WheelLeftSubscriber>(subscriber));
    }
}

void VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::unsubscribeFromAllChanges(IVehicleMotionManagementBrakeAxleRow1WheelLeftSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::subscribeToTorqueChanged(VehicleMotionManagementBrakeAxleRow1WheelLeftTorquePropertyCb callback)
{
    auto handleId = m_torqueChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueCallbacksMutex);
    m_torqueCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::unsubscribeFromTorqueChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueCallbacksMutex);
    m_torqueCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::publishTorqueChanged(int32_t Torque) const
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

long VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::subscribeToTorqueArbitratedChanged(VehicleMotionManagementBrakeAxleRow1WheelLeftTorqueArbitratedPropertyCb callback)
{
    auto handleId = m_torqueArbitratedChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueArbitratedCallbacksMutex);
    m_torqueArbitratedCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::unsubscribeFromTorqueArbitratedChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueArbitratedCallbacksMutex);
    m_torqueArbitratedCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::publishTorqueArbitratedChanged(int32_t TorqueArbitrated) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueArbitratedChanged(TorqueArbitrated);
    }
    std::shared_lock<std::shared_timed_mutex> torqueArbitratedCallbacksLock(m_torqueArbitratedCallbacksMutex);
    const auto torqueArbitratedCallbacks = m_torqueArbitratedCallbacks;
    torqueArbitratedCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueArbitratedCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueArbitrated);
        }
    }
}

long VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::subscribeToTorqueFrictionMaximumChanged(VehicleMotionManagementBrakeAxleRow1WheelLeftTorqueFrictionMaximumPropertyCb callback)
{
    auto handleId = m_torqueFrictionMaximumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionMaximumCallbacksMutex);
    m_torqueFrictionMaximumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::unsubscribeFromTorqueFrictionMaximumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionMaximumCallbacksMutex);
    m_torqueFrictionMaximumCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::publishTorqueFrictionMaximumChanged(int32_t TorqueFrictionMaximum) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueFrictionMaximumChanged(TorqueFrictionMaximum);
    }
    std::shared_lock<std::shared_timed_mutex> torqueFrictionMaximumCallbacksLock(m_torqueFrictionMaximumCallbacksMutex);
    const auto torqueFrictionMaximumCallbacks = m_torqueFrictionMaximumCallbacks;
    torqueFrictionMaximumCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueFrictionMaximumCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueFrictionMaximum);
        }
    }
}

long VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::subscribeToTorqueFrictionMinimumChanged(VehicleMotionManagementBrakeAxleRow1WheelLeftTorqueFrictionMinimumPropertyCb callback)
{
    auto handleId = m_torqueFrictionMinimumChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionMinimumCallbacksMutex);
    m_torqueFrictionMinimumCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::unsubscribeFromTorqueFrictionMinimumChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_torqueFrictionMinimumCallbacksMutex);
    m_torqueFrictionMinimumCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::publishTorqueFrictionMinimumChanged(int32_t TorqueFrictionMinimum) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onTorqueFrictionMinimumChanged(TorqueFrictionMinimum);
    }
    std::shared_lock<std::shared_timed_mutex> torqueFrictionMinimumCallbacksLock(m_torqueFrictionMinimumCallbacksMutex);
    const auto torqueFrictionMinimumCallbacks = m_torqueFrictionMinimumCallbacks;
    torqueFrictionMinimumCallbacksLock.unlock();
    for(const auto& callbackEntry: torqueFrictionMinimumCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(TorqueFrictionMinimum);
        }
    }
}

long VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::subscribeToOmegaUpperChanged(VehicleMotionManagementBrakeAxleRow1WheelLeftOmegaUpperPropertyCb callback)
{
    auto handleId = m_omegaUpperChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_omegaUpperCallbacksMutex);
    m_omegaUpperCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::unsubscribeFromOmegaUpperChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_omegaUpperCallbacksMutex);
    m_omegaUpperCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::publishOmegaUpperChanged(int32_t OmegaUpper) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onOmegaUpperChanged(OmegaUpper);
    }
    std::shared_lock<std::shared_timed_mutex> omegaUpperCallbacksLock(m_omegaUpperCallbacksMutex);
    const auto omegaUpperCallbacks = m_omegaUpperCallbacks;
    omegaUpperCallbacksLock.unlock();
    for(const auto& callbackEntry: omegaUpperCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(OmegaUpper);
        }
    }
}

long VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::subscribeToOmegaLowerChanged(VehicleMotionManagementBrakeAxleRow1WheelLeftOmegaLowerPropertyCb callback)
{
    auto handleId = m_omegaLowerChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_omegaLowerCallbacksMutex);
    m_omegaLowerCallbacks[handleId] = callback;
    return handleId;
}

void VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::unsubscribeFromOmegaLowerChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_omegaLowerCallbacksMutex);
    m_omegaLowerCallbacks.erase(handleId);
}

void VehicleMotionManagementBrakeAxleRow1WheelLeftPublisher::publishOmegaLowerChanged(int32_t OmegaLower) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onOmegaLowerChanged(OmegaLower);
    }
    std::shared_lock<std::shared_timed_mutex> omegaLowerCallbacksLock(m_omegaLowerCallbacksMutex);
    const auto omegaLowerCallbacks = m_omegaLowerCallbacks;
    omegaLowerCallbacksLock.unlock();
    for(const auto& callbackEntry: omegaLowerCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(OmegaLower);
        }
    }
}

