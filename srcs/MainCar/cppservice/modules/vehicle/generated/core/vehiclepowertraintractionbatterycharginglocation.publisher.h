#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertraintractionbatterycharginglocation.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainTractionBatteryChargingLocationPublisher.
 * Use this class to store clients of the Vehicle_Powertrain_TractionBattery_Charging_Location and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingLocationPublisher : public IVehiclePowertrainTractionBatteryChargingLocationPublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingLocationPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainTractionBatteryChargingLocationSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingLocationPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainTractionBatteryChargingLocationSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingLocationPublisher::subscribeToLatitudeChanged
    */
    long subscribeToLatitudeChanged(VehiclePowertrainTractionBatteryChargingLocationLatitudePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingLocationPublisher::subscribeToLatitudeChanged
    */
    void unsubscribeFromLatitudeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingLocationPublisher::subscribeToLongitudeChanged
    */
    long subscribeToLongitudeChanged(VehiclePowertrainTractionBatteryChargingLocationLongitudePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingLocationPublisher::subscribeToLongitudeChanged
    */
    void unsubscribeFromLongitudeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingLocationPublisher::subscribeToAltitudeChanged
    */
    long subscribeToAltitudeChanged(VehiclePowertrainTractionBatteryChargingLocationAltitudePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingLocationPublisher::subscribeToAltitudeChanged
    */
    void unsubscribeFromAltitudeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingLocationPublisher::publishLatitudeChanged
    */
    void publishLatitudeChanged(double Latitude) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingLocationPublisher::publishLongitudeChanged
    */
    void publishLongitudeChanged(double Longitude) const override;
    /**
    * Implementation of IVehiclePowertrainTractionBatteryChargingLocationPublisher::publishAltitudeChanged
    */
    void publishAltitudeChanged(double Altitude) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_TractionBattery_Charging_Location
    std::vector<std::reference_wrapper<IVehiclePowertrainTractionBatteryChargingLocationSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Latitude change.
    std::atomic<long> m_latitudeChangedCallbackNextId {0};
    // Subscribed callbacks for the Latitude change.
    std::map<long, VehiclePowertrainTractionBatteryChargingLocationLatitudePropertyCb> m_latitudeCallbacks;
    // Mutex for m_latitudeCallbacks
    mutable std::shared_timed_mutex m_latitudeCallbacksMutex;
    // Next free unique identifier to subscribe for the Longitude change.
    std::atomic<long> m_longitudeChangedCallbackNextId {0};
    // Subscribed callbacks for the Longitude change.
    std::map<long, VehiclePowertrainTractionBatteryChargingLocationLongitudePropertyCb> m_longitudeCallbacks;
    // Mutex for m_longitudeCallbacks
    mutable std::shared_timed_mutex m_longitudeCallbacksMutex;
    // Next free unique identifier to subscribe for the Altitude change.
    std::atomic<long> m_altitudeChangedCallbackNextId {0};
    // Subscribed callbacks for the Altitude change.
    std::map<long, VehiclePowertrainTractionBatteryChargingLocationAltitudePropertyCb> m_altitudeCallbacks;
    // Mutex for m_altitudeCallbacks
    mutable std::shared_timed_mutex m_altitudeCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
