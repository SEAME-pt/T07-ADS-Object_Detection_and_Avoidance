

#include "vehicle/generated/core/vehiclediagnostics.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleDiagnosticsPublisher::subscribeToAllChanges(IVehicleDiagnosticsSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleDiagnosticsSubscriber>(subscriber));
    }
}

void VehicleDiagnosticsPublisher::unsubscribeFromAllChanges(IVehicleDiagnosticsSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleDiagnosticsPublisher::subscribeToDtcCountChanged(VehicleDiagnosticsDtcCountPropertyCb callback)
{
    auto handleId = m_dtcCountChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_dtcCountCallbacksMutex);
    m_dtcCountCallbacks[handleId] = callback;
    return handleId;
}

void VehicleDiagnosticsPublisher::unsubscribeFromDtcCountChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_dtcCountCallbacksMutex);
    m_dtcCountCallbacks.erase(handleId);
}

void VehicleDiagnosticsPublisher::publishDtcCountChanged(int32_t DTCCount) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onDtcCountChanged(DTCCount);
    }
    std::shared_lock<std::shared_timed_mutex> dtcCountCallbacksLock(m_dtcCountCallbacksMutex);
    const auto dtcCountCallbacks = m_dtcCountCallbacks;
    dtcCountCallbacksLock.unlock();
    for(const auto& callbackEntry: dtcCountCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(DTCCount);
        }
    }
}

long VehicleDiagnosticsPublisher::subscribeToDtcListChanged(VehicleDiagnosticsDtcListPropertyCb callback)
{
    auto handleId = m_dtcListChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_dtcListCallbacksMutex);
    m_dtcListCallbacks[handleId] = callback;
    return handleId;
}

void VehicleDiagnosticsPublisher::unsubscribeFromDtcListChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_dtcListCallbacksMutex);
    m_dtcListCallbacks.erase(handleId);
}

void VehicleDiagnosticsPublisher::publishDtcListChanged(const std::list<std::string>& DTCList) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onDtcListChanged(DTCList);
    }
    std::shared_lock<std::shared_timed_mutex> dtcListCallbacksLock(m_dtcListCallbacksMutex);
    const auto dtcListCallbacks = m_dtcListCallbacks;
    dtcListCallbacksLock.unlock();
    for(const auto& callbackEntry: dtcListCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(DTCList);
        }
    }
}

