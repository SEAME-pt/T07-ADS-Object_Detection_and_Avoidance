#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclebodylightsbackup.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleBodyLightsBackupPublisher.
 * Use this class to store clients of the Vehicle_Body_Lights_Backup and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleBodyLightsBackupPublisher : public IVehicleBodyLightsBackupPublisher
{
public:
    /**
    * Implementation of IVehicleBodyLightsBackupPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleBodyLightsBackupSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleBodyLightsBackupPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleBodyLightsBackupSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleBodyLightsBackupPublisher::subscribeToIsOnChanged
    */
    long subscribeToIsOnChanged(VehicleBodyLightsBackupIsOnPropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyLightsBackupPublisher::subscribeToIsOnChanged
    */
    void unsubscribeFromIsOnChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyLightsBackupPublisher::subscribeToIsDefectChanged
    */
    long subscribeToIsDefectChanged(VehicleBodyLightsBackupIsDefectPropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyLightsBackupPublisher::subscribeToIsDefectChanged
    */
    void unsubscribeFromIsDefectChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyLightsBackupPublisher::publishIsOnChanged
    */
    void publishIsOnChanged(bool IsOn) const override;
    /**
    * Implementation of IVehicleBodyLightsBackupPublisher::publishIsDefectChanged
    */
    void publishIsDefectChanged(bool IsDefect) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Body_Lights_Backup
    std::vector<std::reference_wrapper<IVehicleBodyLightsBackupSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the IsOn change.
    std::atomic<long> m_isOnChangedCallbackNextId {0};
    // Subscribed callbacks for the IsOn change.
    std::map<long, VehicleBodyLightsBackupIsOnPropertyCb> m_isOnCallbacks;
    // Mutex for m_isOnCallbacks
    mutable std::shared_timed_mutex m_isOnCallbacksMutex;
    // Next free unique identifier to subscribe for the IsDefect change.
    std::atomic<long> m_isDefectChangedCallbackNextId {0};
    // Subscribed callbacks for the IsDefect change.
    std::map<long, VehicleBodyLightsBackupIsDefectPropertyCb> m_isDefectCallbacks;
    // Mutex for m_isDefectCallbacks
    mutable std::shared_timed_mutex m_isDefectCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
