#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclebodylights.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleBodyLightsPublisher.
 * Use this class to store clients of the Vehicle_Body_Lights and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleBodyLightsPublisher : public IVehicleBodyLightsPublisher
{
public:
    /**
    * Implementation of IVehicleBodyLightsPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleBodyLightsSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleBodyLightsPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleBodyLightsSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleBodyLightsPublisher::subscribeToLightSwitchChanged
    */
    long subscribeToLightSwitchChanged(VehicleBodyLightsLightSwitchPropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyLightsPublisher::subscribeToLightSwitchChanged
    */
    void unsubscribeFromLightSwitchChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyLightsPublisher::subscribeToIsHighBeamSwitchOnChanged
    */
    long subscribeToIsHighBeamSwitchOnChanged(VehicleBodyLightsIsHighBeamSwitchOnPropertyCb callback) override;
    /**
    * Implementation of IVehicleBodyLightsPublisher::subscribeToIsHighBeamSwitchOnChanged
    */
    void unsubscribeFromIsHighBeamSwitchOnChanged(long handleId) override;

    /**
    * Implementation of IVehicleBodyLightsPublisher::publishLightSwitchChanged
    */
    void publishLightSwitchChanged(Vehicle_Body_Lights_LightSwitch_ValueEnum LightSwitch) const override;
    /**
    * Implementation of IVehicleBodyLightsPublisher::publishIsHighBeamSwitchOnChanged
    */
    void publishIsHighBeamSwitchOnChanged(bool IsHighBeamSwitchOn) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Body_Lights
    std::vector<std::reference_wrapper<IVehicleBodyLightsSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the LightSwitch change.
    std::atomic<long> m_lightSwitchChangedCallbackNextId {0};
    // Subscribed callbacks for the LightSwitch change.
    std::map<long, VehicleBodyLightsLightSwitchPropertyCb> m_lightSwitchCallbacks;
    // Mutex for m_lightSwitchCallbacks
    mutable std::shared_timed_mutex m_lightSwitchCallbacksMutex;
    // Next free unique identifier to subscribe for the IsHighBeamSwitchOn change.
    std::atomic<long> m_isHighBeamSwitchOnChangedCallbackNextId {0};
    // Subscribed callbacks for the IsHighBeamSwitchOn change.
    std::map<long, VehicleBodyLightsIsHighBeamSwitchOnPropertyCb> m_isHighBeamSwitchOnCallbacks;
    // Mutex for m_isHighBeamSwitchOnCallbacks
    mutable std::shared_timed_mutex m_isHighBeamSwitchOnCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
