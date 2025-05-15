#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclecurrentlocation.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleCurrentLocationPublisher.
 * Use this class to store clients of the Vehicle_CurrentLocation and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleCurrentLocationPublisher : public IVehicleCurrentLocationPublisher
{
public:
    /**
    * Implementation of IVehicleCurrentLocationPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleCurrentLocationSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleCurrentLocationPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleCurrentLocationSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleCurrentLocationPublisher::subscribeToTimestampChanged
    */
    long subscribeToTimestampChanged(VehicleCurrentLocationTimestampPropertyCb callback) override;
    /**
    * Implementation of IVehicleCurrentLocationPublisher::subscribeToTimestampChanged
    */
    void unsubscribeFromTimestampChanged(long handleId) override;

    /**
    * Implementation of IVehicleCurrentLocationPublisher::subscribeToLatitudeChanged
    */
    long subscribeToLatitudeChanged(VehicleCurrentLocationLatitudePropertyCb callback) override;
    /**
    * Implementation of IVehicleCurrentLocationPublisher::subscribeToLatitudeChanged
    */
    void unsubscribeFromLatitudeChanged(long handleId) override;

    /**
    * Implementation of IVehicleCurrentLocationPublisher::subscribeToLongitudeChanged
    */
    long subscribeToLongitudeChanged(VehicleCurrentLocationLongitudePropertyCb callback) override;
    /**
    * Implementation of IVehicleCurrentLocationPublisher::subscribeToLongitudeChanged
    */
    void unsubscribeFromLongitudeChanged(long handleId) override;

    /**
    * Implementation of IVehicleCurrentLocationPublisher::subscribeToHeadingChanged
    */
    long subscribeToHeadingChanged(VehicleCurrentLocationHeadingPropertyCb callback) override;
    /**
    * Implementation of IVehicleCurrentLocationPublisher::subscribeToHeadingChanged
    */
    void unsubscribeFromHeadingChanged(long handleId) override;

    /**
    * Implementation of IVehicleCurrentLocationPublisher::subscribeToHorizontalAccuracyChanged
    */
    long subscribeToHorizontalAccuracyChanged(VehicleCurrentLocationHorizontalAccuracyPropertyCb callback) override;
    /**
    * Implementation of IVehicleCurrentLocationPublisher::subscribeToHorizontalAccuracyChanged
    */
    void unsubscribeFromHorizontalAccuracyChanged(long handleId) override;

    /**
    * Implementation of IVehicleCurrentLocationPublisher::subscribeToAltitudeChanged
    */
    long subscribeToAltitudeChanged(VehicleCurrentLocationAltitudePropertyCb callback) override;
    /**
    * Implementation of IVehicleCurrentLocationPublisher::subscribeToAltitudeChanged
    */
    void unsubscribeFromAltitudeChanged(long handleId) override;

    /**
    * Implementation of IVehicleCurrentLocationPublisher::subscribeToVerticalAccuracyChanged
    */
    long subscribeToVerticalAccuracyChanged(VehicleCurrentLocationVerticalAccuracyPropertyCb callback) override;
    /**
    * Implementation of IVehicleCurrentLocationPublisher::subscribeToVerticalAccuracyChanged
    */
    void unsubscribeFromVerticalAccuracyChanged(long handleId) override;

    /**
    * Implementation of IVehicleCurrentLocationPublisher::publishTimestampChanged
    */
    void publishTimestampChanged(const std::string& Timestamp) const override;
    /**
    * Implementation of IVehicleCurrentLocationPublisher::publishLatitudeChanged
    */
    void publishLatitudeChanged(double Latitude) const override;
    /**
    * Implementation of IVehicleCurrentLocationPublisher::publishLongitudeChanged
    */
    void publishLongitudeChanged(double Longitude) const override;
    /**
    * Implementation of IVehicleCurrentLocationPublisher::publishHeadingChanged
    */
    void publishHeadingChanged(double Heading) const override;
    /**
    * Implementation of IVehicleCurrentLocationPublisher::publishHorizontalAccuracyChanged
    */
    void publishHorizontalAccuracyChanged(double HorizontalAccuracy) const override;
    /**
    * Implementation of IVehicleCurrentLocationPublisher::publishAltitudeChanged
    */
    void publishAltitudeChanged(double Altitude) const override;
    /**
    * Implementation of IVehicleCurrentLocationPublisher::publishVerticalAccuracyChanged
    */
    void publishVerticalAccuracyChanged(double VerticalAccuracy) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_CurrentLocation
    std::vector<std::reference_wrapper<IVehicleCurrentLocationSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Timestamp change.
    std::atomic<long> m_timestampChangedCallbackNextId {0};
    // Subscribed callbacks for the Timestamp change.
    std::map<long, VehicleCurrentLocationTimestampPropertyCb> m_timestampCallbacks;
    // Mutex for m_timestampCallbacks
    mutable std::shared_timed_mutex m_timestampCallbacksMutex;
    // Next free unique identifier to subscribe for the Latitude change.
    std::atomic<long> m_latitudeChangedCallbackNextId {0};
    // Subscribed callbacks for the Latitude change.
    std::map<long, VehicleCurrentLocationLatitudePropertyCb> m_latitudeCallbacks;
    // Mutex for m_latitudeCallbacks
    mutable std::shared_timed_mutex m_latitudeCallbacksMutex;
    // Next free unique identifier to subscribe for the Longitude change.
    std::atomic<long> m_longitudeChangedCallbackNextId {0};
    // Subscribed callbacks for the Longitude change.
    std::map<long, VehicleCurrentLocationLongitudePropertyCb> m_longitudeCallbacks;
    // Mutex for m_longitudeCallbacks
    mutable std::shared_timed_mutex m_longitudeCallbacksMutex;
    // Next free unique identifier to subscribe for the Heading change.
    std::atomic<long> m_headingChangedCallbackNextId {0};
    // Subscribed callbacks for the Heading change.
    std::map<long, VehicleCurrentLocationHeadingPropertyCb> m_headingCallbacks;
    // Mutex for m_headingCallbacks
    mutable std::shared_timed_mutex m_headingCallbacksMutex;
    // Next free unique identifier to subscribe for the HorizontalAccuracy change.
    std::atomic<long> m_horizontalAccuracyChangedCallbackNextId {0};
    // Subscribed callbacks for the HorizontalAccuracy change.
    std::map<long, VehicleCurrentLocationHorizontalAccuracyPropertyCb> m_horizontalAccuracyCallbacks;
    // Mutex for m_horizontalAccuracyCallbacks
    mutable std::shared_timed_mutex m_horizontalAccuracyCallbacksMutex;
    // Next free unique identifier to subscribe for the Altitude change.
    std::atomic<long> m_altitudeChangedCallbackNextId {0};
    // Subscribed callbacks for the Altitude change.
    std::map<long, VehicleCurrentLocationAltitudePropertyCb> m_altitudeCallbacks;
    // Mutex for m_altitudeCallbacks
    mutable std::shared_timed_mutex m_altitudeCallbacksMutex;
    // Next free unique identifier to subscribe for the VerticalAccuracy change.
    std::atomic<long> m_verticalAccuracyChangedCallbackNextId {0};
    // Subscribed callbacks for the VerticalAccuracy change.
    std::map<long, VehicleCurrentLocationVerticalAccuracyPropertyCb> m_verticalAccuracyCallbacks;
    // Mutex for m_verticalAccuracyCallbacks
    mutable std::shared_timed_mutex m_verticalAccuracyCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
