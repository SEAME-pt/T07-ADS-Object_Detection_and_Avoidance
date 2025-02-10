

#include "vehicle/generated/core/vehicleversionvss.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleVersionVssPublisher::subscribeToAllChanges(IVehicleVersionVssSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleVersionVssSubscriber>(subscriber));
    }
}

void VehicleVersionVssPublisher::unsubscribeFromAllChanges(IVehicleVersionVssSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleVersionVssPublisher::subscribeToMajorChanged(VehicleVersionVssMajorPropertyCb callback)
{
    auto handleId = m_majorChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_majorCallbacksMutex);
    m_majorCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVersionVssPublisher::unsubscribeFromMajorChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_majorCallbacksMutex);
    m_majorCallbacks.erase(handleId);
}

void VehicleVersionVssPublisher::publishMajorChanged(int32_t Major) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onMajorChanged(Major);
    }
    std::shared_lock<std::shared_timed_mutex> majorCallbacksLock(m_majorCallbacksMutex);
    const auto majorCallbacks = m_majorCallbacks;
    majorCallbacksLock.unlock();
    for(const auto& callbackEntry: majorCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Major);
        }
    }
}

long VehicleVersionVssPublisher::subscribeToMinorChanged(VehicleVersionVssMinorPropertyCb callback)
{
    auto handleId = m_minorChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_minorCallbacksMutex);
    m_minorCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVersionVssPublisher::unsubscribeFromMinorChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_minorCallbacksMutex);
    m_minorCallbacks.erase(handleId);
}

void VehicleVersionVssPublisher::publishMinorChanged(int32_t Minor) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onMinorChanged(Minor);
    }
    std::shared_lock<std::shared_timed_mutex> minorCallbacksLock(m_minorCallbacksMutex);
    const auto minorCallbacks = m_minorCallbacks;
    minorCallbacksLock.unlock();
    for(const auto& callbackEntry: minorCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Minor);
        }
    }
}

long VehicleVersionVssPublisher::subscribeToPatchChanged(VehicleVersionVssPatchPropertyCb callback)
{
    auto handleId = m_patchChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_patchCallbacksMutex);
    m_patchCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVersionVssPublisher::unsubscribeFromPatchChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_patchCallbacksMutex);
    m_patchCallbacks.erase(handleId);
}

void VehicleVersionVssPublisher::publishPatchChanged(int32_t Patch) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onPatchChanged(Patch);
    }
    std::shared_lock<std::shared_timed_mutex> patchCallbacksLock(m_patchCallbacksMutex);
    const auto patchCallbacks = m_patchCallbacks;
    patchCallbacksLock.unlock();
    for(const auto& callbackEntry: patchCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Patch);
        }
    }
}

long VehicleVersionVssPublisher::subscribeToLabelChanged(VehicleVersionVssLabelPropertyCb callback)
{
    auto handleId = m_labelChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_labelCallbacksMutex);
    m_labelCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVersionVssPublisher::unsubscribeFromLabelChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_labelCallbacksMutex);
    m_labelCallbacks.erase(handleId);
}

void VehicleVersionVssPublisher::publishLabelChanged(const std::string& Label) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onLabelChanged(Label);
    }
    std::shared_lock<std::shared_timed_mutex> labelCallbacksLock(m_labelCallbacksMutex);
    const auto labelCallbacks = m_labelCallbacks;
    labelCallbacksLock.unlock();
    for(const auto& callbackEntry: labelCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Label);
        }
    }
}

