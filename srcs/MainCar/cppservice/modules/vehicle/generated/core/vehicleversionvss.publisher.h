#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehicleversionvss.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleVersionVssPublisher.
 * Use this class to store clients of the Vehicle_VersionVSS and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleVersionVssPublisher : public IVehicleVersionVssPublisher
{
public:
    /**
    * Implementation of IVehicleVersionVssPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleVersionVssSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleVersionVssPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleVersionVssSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleVersionVssPublisher::subscribeToMajorChanged
    */
    long subscribeToMajorChanged(VehicleVersionVssMajorPropertyCb callback) override;
    /**
    * Implementation of IVehicleVersionVssPublisher::subscribeToMajorChanged
    */
    void unsubscribeFromMajorChanged(long handleId) override;

    /**
    * Implementation of IVehicleVersionVssPublisher::subscribeToMinorChanged
    */
    long subscribeToMinorChanged(VehicleVersionVssMinorPropertyCb callback) override;
    /**
    * Implementation of IVehicleVersionVssPublisher::subscribeToMinorChanged
    */
    void unsubscribeFromMinorChanged(long handleId) override;

    /**
    * Implementation of IVehicleVersionVssPublisher::subscribeToPatchChanged
    */
    long subscribeToPatchChanged(VehicleVersionVssPatchPropertyCb callback) override;
    /**
    * Implementation of IVehicleVersionVssPublisher::subscribeToPatchChanged
    */
    void unsubscribeFromPatchChanged(long handleId) override;

    /**
    * Implementation of IVehicleVersionVssPublisher::subscribeToLabelChanged
    */
    long subscribeToLabelChanged(VehicleVersionVssLabelPropertyCb callback) override;
    /**
    * Implementation of IVehicleVersionVssPublisher::subscribeToLabelChanged
    */
    void unsubscribeFromLabelChanged(long handleId) override;

    /**
    * Implementation of IVehicleVersionVssPublisher::publishMajorChanged
    */
    void publishMajorChanged(int32_t Major) const override;
    /**
    * Implementation of IVehicleVersionVssPublisher::publishMinorChanged
    */
    void publishMinorChanged(int32_t Minor) const override;
    /**
    * Implementation of IVehicleVersionVssPublisher::publishPatchChanged
    */
    void publishPatchChanged(int32_t Patch) const override;
    /**
    * Implementation of IVehicleVersionVssPublisher::publishLabelChanged
    */
    void publishLabelChanged(const std::string& Label) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_VersionVSS
    std::vector<std::reference_wrapper<IVehicleVersionVssSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Major change.
    std::atomic<long> m_majorChangedCallbackNextId {0};
    // Subscribed callbacks for the Major change.
    std::map<long, VehicleVersionVssMajorPropertyCb> m_majorCallbacks;
    // Mutex for m_majorCallbacks
    mutable std::shared_timed_mutex m_majorCallbacksMutex;
    // Next free unique identifier to subscribe for the Minor change.
    std::atomic<long> m_minorChangedCallbackNextId {0};
    // Subscribed callbacks for the Minor change.
    std::map<long, VehicleVersionVssMinorPropertyCb> m_minorCallbacks;
    // Mutex for m_minorCallbacks
    mutable std::shared_timed_mutex m_minorCallbacksMutex;
    // Next free unique identifier to subscribe for the Patch change.
    std::atomic<long> m_patchChangedCallbackNextId {0};
    // Subscribed callbacks for the Patch change.
    std::map<long, VehicleVersionVssPatchPropertyCb> m_patchCallbacks;
    // Mutex for m_patchCallbacks
    mutable std::shared_timed_mutex m_patchCallbacksMutex;
    // Next free unique identifier to subscribe for the Label change.
    std::atomic<long> m_labelChangedCallbackNextId {0};
    // Subscribed callbacks for the Label change.
    std::map<long, VehicleVersionVssLabelPropertyCb> m_labelCallbacks;
    // Mutex for m_labelCallbacks
    mutable std::shared_timed_mutex m_labelCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
