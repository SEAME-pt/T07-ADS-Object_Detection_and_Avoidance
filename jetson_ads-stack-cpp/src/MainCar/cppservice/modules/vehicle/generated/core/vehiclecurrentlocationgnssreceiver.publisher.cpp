

#include "vehicle/generated/core/vehiclecurrentlocationgnssreceiver.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleCurrentLocationGnssReceiverPublisher::subscribeToAllChanges(IVehicleCurrentLocationGnssReceiverSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleCurrentLocationGnssReceiverSubscriber>(subscriber));
    }
}

void VehicleCurrentLocationGnssReceiverPublisher::unsubscribeFromAllChanges(IVehicleCurrentLocationGnssReceiverSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleCurrentLocationGnssReceiverPublisher::subscribeToFixTypeChanged(VehicleCurrentLocationGnssReceiverFixTypePropertyCb callback)
{
    auto handleId = m_fixTypeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_fixTypeCallbacksMutex);
    m_fixTypeCallbacks[handleId] = callback;
    return handleId;
}

void VehicleCurrentLocationGnssReceiverPublisher::unsubscribeFromFixTypeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_fixTypeCallbacksMutex);
    m_fixTypeCallbacks.erase(handleId);
}

void VehicleCurrentLocationGnssReceiverPublisher::publishFixTypeChanged(Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum FixType) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onFixTypeChanged(FixType);
    }
    std::shared_lock<std::shared_timed_mutex> fixTypeCallbacksLock(m_fixTypeCallbacksMutex);
    const auto fixTypeCallbacks = m_fixTypeCallbacks;
    fixTypeCallbacksLock.unlock();
    for(const auto& callbackEntry: fixTypeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(FixType);
        }
    }
}

