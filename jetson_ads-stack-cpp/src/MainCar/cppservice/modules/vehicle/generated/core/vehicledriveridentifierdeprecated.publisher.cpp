

#include "vehicle/generated/core/vehicledriveridentifierdeprecated.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleDriverIdentifierDeprecatedPublisher::subscribeToAllChanges(IVehicleDriverIdentifierDeprecatedSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleDriverIdentifierDeprecatedSubscriber>(subscriber));
    }
}

void VehicleDriverIdentifierDeprecatedPublisher::unsubscribeFromAllChanges(IVehicleDriverIdentifierDeprecatedSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleDriverIdentifierDeprecatedPublisher::subscribeToSubjectDeprecatedChanged(VehicleDriverIdentifierDeprecatedSubjectDeprecatedPropertyCb callback)
{
    auto handleId = m_subjectDeprecatedChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_subjectDeprecatedCallbacksMutex);
    m_subjectDeprecatedCallbacks[handleId] = callback;
    return handleId;
}

void VehicleDriverIdentifierDeprecatedPublisher::unsubscribeFromSubjectDeprecatedChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_subjectDeprecatedCallbacksMutex);
    m_subjectDeprecatedCallbacks.erase(handleId);
}

void VehicleDriverIdentifierDeprecatedPublisher::publishSubjectDeprecatedChanged(const std::string& Subject_Deprecated) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onSubjectDeprecatedChanged(Subject_Deprecated);
    }
    std::shared_lock<std::shared_timed_mutex> subjectDeprecatedCallbacksLock(m_subjectDeprecatedCallbacksMutex);
    const auto subjectDeprecatedCallbacks = m_subjectDeprecatedCallbacks;
    subjectDeprecatedCallbacksLock.unlock();
    for(const auto& callbackEntry: subjectDeprecatedCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Subject_Deprecated);
        }
    }
}

long VehicleDriverIdentifierDeprecatedPublisher::subscribeToIssuerDeprecatedChanged(VehicleDriverIdentifierDeprecatedIssuerDeprecatedPropertyCb callback)
{
    auto handleId = m_issuerDeprecatedChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_issuerDeprecatedCallbacksMutex);
    m_issuerDeprecatedCallbacks[handleId] = callback;
    return handleId;
}

void VehicleDriverIdentifierDeprecatedPublisher::unsubscribeFromIssuerDeprecatedChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_issuerDeprecatedCallbacksMutex);
    m_issuerDeprecatedCallbacks.erase(handleId);
}

void VehicleDriverIdentifierDeprecatedPublisher::publishIssuerDeprecatedChanged(const std::string& Issuer_Deprecated) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onIssuerDeprecatedChanged(Issuer_Deprecated);
    }
    std::shared_lock<std::shared_timed_mutex> issuerDeprecatedCallbacksLock(m_issuerDeprecatedCallbacksMutex);
    const auto issuerDeprecatedCallbacks = m_issuerDeprecatedCallbacks;
    issuerDeprecatedCallbacksLock.unlock();
    for(const auto& callbackEntry: issuerDeprecatedCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Issuer_Deprecated);
        }
    }
}

