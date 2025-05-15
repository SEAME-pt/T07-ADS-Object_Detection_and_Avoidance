#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclediagnostics.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleDiagnosticsPublisher.
 * Use this class to store clients of the Vehicle_Diagnostics and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleDiagnosticsPublisher : public IVehicleDiagnosticsPublisher
{
public:
    /**
    * Implementation of IVehicleDiagnosticsPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleDiagnosticsSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleDiagnosticsPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleDiagnosticsSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleDiagnosticsPublisher::subscribeToDtcCountChanged
    */
    long subscribeToDtcCountChanged(VehicleDiagnosticsDtcCountPropertyCb callback) override;
    /**
    * Implementation of IVehicleDiagnosticsPublisher::subscribeToDtcCountChanged
    */
    void unsubscribeFromDtcCountChanged(long handleId) override;

    /**
    * Implementation of IVehicleDiagnosticsPublisher::subscribeToDtcListChanged
    */
    long subscribeToDtcListChanged(VehicleDiagnosticsDtcListPropertyCb callback) override;
    /**
    * Implementation of IVehicleDiagnosticsPublisher::subscribeToDtcListChanged
    */
    void unsubscribeFromDtcListChanged(long handleId) override;

    /**
    * Implementation of IVehicleDiagnosticsPublisher::publishDtcCountChanged
    */
    void publishDtcCountChanged(int32_t DTCCount) const override;
    /**
    * Implementation of IVehicleDiagnosticsPublisher::publishDtcListChanged
    */
    void publishDtcListChanged(const std::list<std::string>& DTCList) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Diagnostics
    std::vector<std::reference_wrapper<IVehicleDiagnosticsSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the DtcCount change.
    std::atomic<long> m_dtcCountChangedCallbackNextId {0};
    // Subscribed callbacks for the DtcCount change.
    std::map<long, VehicleDiagnosticsDtcCountPropertyCb> m_dtcCountCallbacks;
    // Mutex for m_dtcCountCallbacks
    mutable std::shared_timed_mutex m_dtcCountCallbacksMutex;
    // Next free unique identifier to subscribe for the DtcList change.
    std::atomic<long> m_dtcListChangedCallbackNextId {0};
    // Subscribed callbacks for the DtcList change.
    std::map<long, VehicleDiagnosticsDtcListPropertyCb> m_dtcListCallbacks;
    // Mutex for m_dtcListCallbacks
    mutable std::shared_timed_mutex m_dtcListCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
