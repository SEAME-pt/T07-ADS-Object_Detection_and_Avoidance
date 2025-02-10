#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclebodylightsbrake.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleBodyLightsBrakePublisher.
 * Use this class to store clients of the Vehicle_Body_Lights_Brake and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleBodyLightsBrakePublisher : public IVehicleBodyLightsBrakePublisher
{
public:
    /**
    * Implementation of IVehicleBodyLightsBrakePublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleBodyLightsBrakeSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleBodyLightsBrakePublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleBodyLightsBrakeSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleBodyLightsBrakePublisher::subscribeToIsActiveChanged
    */
    long subscribeToIsActiveChanged(VehicleBodyLightsBrakeIsActivePropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyLightsBrakePublisher::subscribeToIsActiveChanged
    */
    void unsubscribeFromIsActiveChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyLightsBrakePublisher::subscribeToIsDefectChanged
    */
    long subscribeToIsDefectChanged(VehicleBodyLightsBrakeIsDefectPropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyLightsBrakePublisher::subscribeToIsDefectChanged
    */
    void unsubscribeFromIsDefectChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyLightsBrakePublisher::publishIsActiveChanged
    */
    void publishIsActiveChanged(Vehicle_Body_Lights_Brake_IsActive_ValueEnum IsActive) const override;
    /**
    * Implementation of IVehicleBodyLightsBrakePublisher::publishIsDefectChanged
    */
    void publishIsDefectChanged(bool IsDefect) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Body_Lights_Brake
    std::vector<std::reference_wrapper<IVehicleBodyLightsBrakeSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the IsActive change.
    std::atomic<long> m_isActiveChangedCallbackNextId {0};
    // Subscribed callbacks for the IsActive change.
    std::map<long, VehicleBodyLightsBrakeIsActivePropertyCb> m_isActiveCallbacks;
    // Mutex for m_isActiveCallbacks
    mutable std::shared_timed_mutex m_isActiveCallbacksMutex;
    // Next free unique identifier to subscribe for the IsDefect change.
    std::atomic<long> m_isDefectChangedCallbackNextId {0};
    // Subscribed callbacks for the IsDefect change.
    std::map<long, VehicleBodyLightsBrakeIsDefectPropertyCb> m_isDefectCallbacks;
    // Mutex for m_isDefectCallbacks
    mutable std::shared_timed_mutex m_isDefectCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
