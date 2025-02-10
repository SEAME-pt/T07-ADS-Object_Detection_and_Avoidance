#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehicledriveridentifierdeprecated.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleDriverIdentifierDeprecatedPublisher.
 * Use this class to store clients of the Vehicle_Driver_Identifier_Deprecated and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleDriverIdentifierDeprecatedPublisher : public IVehicleDriverIdentifierDeprecatedPublisher
{
public:
    /**
    * Implementation of IVehicleDriverIdentifierDeprecatedPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleDriverIdentifierDeprecatedSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleDriverIdentifierDeprecatedPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleDriverIdentifierDeprecatedSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleDriverIdentifierDeprecatedPublisher::subscribeToSubjectDeprecatedChanged
    */
    long subscribeToSubjectDeprecatedChanged(VehicleDriverIdentifierDeprecatedSubjectDeprecatedPropertyCb callback) override;
    /**
    * Implementation of IVehicleDriverIdentifierDeprecatedPublisher::subscribeToSubjectDeprecatedChanged
    */
    void unsubscribeFromSubjectDeprecatedChanged(long handleId) override;

    /**
    * Implementation of IVehicleDriverIdentifierDeprecatedPublisher::subscribeToIssuerDeprecatedChanged
    */
    long subscribeToIssuerDeprecatedChanged(VehicleDriverIdentifierDeprecatedIssuerDeprecatedPropertyCb callback) override;
    /**
    * Implementation of IVehicleDriverIdentifierDeprecatedPublisher::subscribeToIssuerDeprecatedChanged
    */
    void unsubscribeFromIssuerDeprecatedChanged(long handleId) override;

    /**
    * Implementation of IVehicleDriverIdentifierDeprecatedPublisher::publishSubjectDeprecatedChanged
    */
    void publishSubjectDeprecatedChanged(const std::string& Subject_Deprecated) const override;
    /**
    * Implementation of IVehicleDriverIdentifierDeprecatedPublisher::publishIssuerDeprecatedChanged
    */
    void publishIssuerDeprecatedChanged(const std::string& Issuer_Deprecated) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Driver_Identifier_Deprecated
    std::vector<std::reference_wrapper<IVehicleDriverIdentifierDeprecatedSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the SubjectDeprecated change.
    std::atomic<long> m_subjectDeprecatedChangedCallbackNextId {0};
    // Subscribed callbacks for the SubjectDeprecated change.
    std::map<long, VehicleDriverIdentifierDeprecatedSubjectDeprecatedPropertyCb> m_subjectDeprecatedCallbacks;
    // Mutex for m_subjectDeprecatedCallbacks
    mutable std::shared_timed_mutex m_subjectDeprecatedCallbacksMutex;
    // Next free unique identifier to subscribe for the IssuerDeprecated change.
    std::atomic<long> m_issuerDeprecatedChangedCallbackNextId {0};
    // Subscribed callbacks for the IssuerDeprecated change.
    std::map<long, VehicleDriverIdentifierDeprecatedIssuerDeprecatedPropertyCb> m_issuerDeprecatedCallbacks;
    // Mutex for m_issuerDeprecatedCallbacks
    mutable std::shared_timed_mutex m_issuerDeprecatedCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
