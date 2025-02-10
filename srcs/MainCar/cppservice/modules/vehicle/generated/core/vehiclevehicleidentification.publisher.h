#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclevehicleidentification.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehicleVehicleIdentificationPublisher.
 * Use this class to store clients of the Vehicle_VehicleIdentification and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehicleVehicleIdentificationPublisher : public IVehicleVehicleIdentificationPublisher
{
public:
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehicleVehicleIdentificationSubscriber& subscriber) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehicleVehicleIdentificationSubscriber& subscriber) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToVinChanged
    */
    long subscribeToVinChanged(VehicleVehicleIdentificationVinPropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToVinChanged
    */
    void unsubscribeFromVinChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToWmiChanged
    */
    long subscribeToWmiChanged(VehicleVehicleIdentificationWmiPropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToWmiChanged
    */
    void unsubscribeFromWmiChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToBrandChanged
    */
    long subscribeToBrandChanged(VehicleVehicleIdentificationBrandPropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToBrandChanged
    */
    void unsubscribeFromBrandChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToModelChanged
    */
    long subscribeToModelChanged(VehicleVehicleIdentificationModelPropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToModelChanged
    */
    void unsubscribeFromModelChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToYearChanged
    */
    long subscribeToYearChanged(VehicleVehicleIdentificationYearPropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToYearChanged
    */
    void unsubscribeFromYearChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToAcrissCodeChanged
    */
    long subscribeToAcrissCodeChanged(VehicleVehicleIdentificationAcrissCodePropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToAcrissCodeChanged
    */
    void unsubscribeFromAcrissCodeChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToBodyTypeChanged
    */
    long subscribeToBodyTypeChanged(VehicleVehicleIdentificationBodyTypePropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToBodyTypeChanged
    */
    void unsubscribeFromBodyTypeChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToDateVehicleFirstRegisteredChanged
    */
    long subscribeToDateVehicleFirstRegisteredChanged(VehicleVehicleIdentificationDateVehicleFirstRegisteredPropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToDateVehicleFirstRegisteredChanged
    */
    void unsubscribeFromDateVehicleFirstRegisteredChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToLicensePlateChanged
    */
    long subscribeToLicensePlateChanged(VehicleVehicleIdentificationLicensePlatePropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToLicensePlateChanged
    */
    void unsubscribeFromLicensePlateChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToMeetsEmissionStandardChanged
    */
    long subscribeToMeetsEmissionStandardChanged(VehicleVehicleIdentificationMeetsEmissionStandardPropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToMeetsEmissionStandardChanged
    */
    void unsubscribeFromMeetsEmissionStandardChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToProductionDateChanged
    */
    long subscribeToProductionDateChanged(VehicleVehicleIdentificationProductionDatePropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToProductionDateChanged
    */
    void unsubscribeFromProductionDateChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToPurchaseDateChanged
    */
    long subscribeToPurchaseDateChanged(VehicleVehicleIdentificationPurchaseDatePropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToPurchaseDateChanged
    */
    void unsubscribeFromPurchaseDateChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToVehicleModelDateChanged
    */
    long subscribeToVehicleModelDateChanged(VehicleVehicleIdentificationVehicleModelDatePropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToVehicleModelDateChanged
    */
    void unsubscribeFromVehicleModelDateChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToVehicleConfigurationChanged
    */
    long subscribeToVehicleConfigurationChanged(VehicleVehicleIdentificationVehicleConfigurationPropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToVehicleConfigurationChanged
    */
    void unsubscribeFromVehicleConfigurationChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToVehicleSeatingCapacityChanged
    */
    long subscribeToVehicleSeatingCapacityChanged(VehicleVehicleIdentificationVehicleSeatingCapacityPropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToVehicleSeatingCapacityChanged
    */
    void unsubscribeFromVehicleSeatingCapacityChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToVehicleSpecialUsageChanged
    */
    long subscribeToVehicleSpecialUsageChanged(VehicleVehicleIdentificationVehicleSpecialUsagePropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToVehicleSpecialUsageChanged
    */
    void unsubscribeFromVehicleSpecialUsageChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToVehicleExteriorColorChanged
    */
    long subscribeToVehicleExteriorColorChanged(VehicleVehicleIdentificationVehicleExteriorColorPropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToVehicleExteriorColorChanged
    */
    void unsubscribeFromVehicleExteriorColorChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToVehicleInteriorColorChanged
    */
    long subscribeToVehicleInteriorColorChanged(VehicleVehicleIdentificationVehicleInteriorColorPropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToVehicleInteriorColorChanged
    */
    void unsubscribeFromVehicleInteriorColorChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToVehicleInteriorTypeChanged
    */
    long subscribeToVehicleInteriorTypeChanged(VehicleVehicleIdentificationVehicleInteriorTypePropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToVehicleInteriorTypeChanged
    */
    void unsubscribeFromVehicleInteriorTypeChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToKnownVehicleDamagesChanged
    */
    long subscribeToKnownVehicleDamagesChanged(VehicleVehicleIdentificationKnownVehicleDamagesPropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToKnownVehicleDamagesChanged
    */
    void unsubscribeFromKnownVehicleDamagesChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToOptionalExtrasChanged
    */
    long subscribeToOptionalExtrasChanged(VehicleVehicleIdentificationOptionalExtrasPropertyCb callback) override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::subscribeToOptionalExtrasChanged
    */
    void unsubscribeFromOptionalExtrasChanged(long handleId) override;

    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishVinChanged
    */
    void publishVinChanged(const std::string& VIN) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishWmiChanged
    */
    void publishWmiChanged(const std::string& WMI) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishBrandChanged
    */
    void publishBrandChanged(const std::string& Brand) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishModelChanged
    */
    void publishModelChanged(const std::string& Model) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishYearChanged
    */
    void publishYearChanged(int32_t Year) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishAcrissCodeChanged
    */
    void publishAcrissCodeChanged(const std::string& AcrissCode) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishBodyTypeChanged
    */
    void publishBodyTypeChanged(const std::string& BodyType) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishDateVehicleFirstRegisteredChanged
    */
    void publishDateVehicleFirstRegisteredChanged(const std::string& DateVehicleFirstRegistered) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishLicensePlateChanged
    */
    void publishLicensePlateChanged(const std::string& LicensePlate) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishMeetsEmissionStandardChanged
    */
    void publishMeetsEmissionStandardChanged(const std::string& MeetsEmissionStandard) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishProductionDateChanged
    */
    void publishProductionDateChanged(const std::string& ProductionDate) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishPurchaseDateChanged
    */
    void publishPurchaseDateChanged(const std::string& PurchaseDate) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishVehicleModelDateChanged
    */
    void publishVehicleModelDateChanged(const std::string& VehicleModelDate) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishVehicleConfigurationChanged
    */
    void publishVehicleConfigurationChanged(const std::string& VehicleConfiguration) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishVehicleSeatingCapacityChanged
    */
    void publishVehicleSeatingCapacityChanged(int32_t VehicleSeatingCapacity) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishVehicleSpecialUsageChanged
    */
    void publishVehicleSpecialUsageChanged(const std::string& VehicleSpecialUsage) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishVehicleExteriorColorChanged
    */
    void publishVehicleExteriorColorChanged(const std::string& VehicleExteriorColor) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishVehicleInteriorColorChanged
    */
    void publishVehicleInteriorColorChanged(const std::string& VehicleInteriorColor) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishVehicleInteriorTypeChanged
    */
    void publishVehicleInteriorTypeChanged(const std::string& VehicleInteriorType) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishKnownVehicleDamagesChanged
    */
    void publishKnownVehicleDamagesChanged(const std::string& KnownVehicleDamages) const override;
    /**
    * Implementation of IVehicleVehicleIdentificationPublisher::publishOptionalExtrasChanged
    */
    void publishOptionalExtrasChanged(const std::list<std::string>& OptionalExtras) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_VehicleIdentification
    std::vector<std::reference_wrapper<IVehicleVehicleIdentificationSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Vin change.
    std::atomic<long> m_vinChangedCallbackNextId {0};
    // Subscribed callbacks for the Vin change.
    std::map<long, VehicleVehicleIdentificationVinPropertyCb> m_vinCallbacks;
    // Mutex for m_vinCallbacks
    mutable std::shared_timed_mutex m_vinCallbacksMutex;
    // Next free unique identifier to subscribe for the Wmi change.
    std::atomic<long> m_wmiChangedCallbackNextId {0};
    // Subscribed callbacks for the Wmi change.
    std::map<long, VehicleVehicleIdentificationWmiPropertyCb> m_wmiCallbacks;
    // Mutex for m_wmiCallbacks
    mutable std::shared_timed_mutex m_wmiCallbacksMutex;
    // Next free unique identifier to subscribe for the Brand change.
    std::atomic<long> m_brandChangedCallbackNextId {0};
    // Subscribed callbacks for the Brand change.
    std::map<long, VehicleVehicleIdentificationBrandPropertyCb> m_brandCallbacks;
    // Mutex for m_brandCallbacks
    mutable std::shared_timed_mutex m_brandCallbacksMutex;
    // Next free unique identifier to subscribe for the Model change.
    std::atomic<long> m_modelChangedCallbackNextId {0};
    // Subscribed callbacks for the Model change.
    std::map<long, VehicleVehicleIdentificationModelPropertyCb> m_modelCallbacks;
    // Mutex for m_modelCallbacks
    mutable std::shared_timed_mutex m_modelCallbacksMutex;
    // Next free unique identifier to subscribe for the Year change.
    std::atomic<long> m_yearChangedCallbackNextId {0};
    // Subscribed callbacks for the Year change.
    std::map<long, VehicleVehicleIdentificationYearPropertyCb> m_yearCallbacks;
    // Mutex for m_yearCallbacks
    mutable std::shared_timed_mutex m_yearCallbacksMutex;
    // Next free unique identifier to subscribe for the AcrissCode change.
    std::atomic<long> m_acrissCodeChangedCallbackNextId {0};
    // Subscribed callbacks for the AcrissCode change.
    std::map<long, VehicleVehicleIdentificationAcrissCodePropertyCb> m_acrissCodeCallbacks;
    // Mutex for m_acrissCodeCallbacks
    mutable std::shared_timed_mutex m_acrissCodeCallbacksMutex;
    // Next free unique identifier to subscribe for the BodyType change.
    std::atomic<long> m_bodyTypeChangedCallbackNextId {0};
    // Subscribed callbacks for the BodyType change.
    std::map<long, VehicleVehicleIdentificationBodyTypePropertyCb> m_bodyTypeCallbacks;
    // Mutex for m_bodyTypeCallbacks
    mutable std::shared_timed_mutex m_bodyTypeCallbacksMutex;
    // Next free unique identifier to subscribe for the DateVehicleFirstRegistered change.
    std::atomic<long> m_dateVehicleFirstRegisteredChangedCallbackNextId {0};
    // Subscribed callbacks for the DateVehicleFirstRegistered change.
    std::map<long, VehicleVehicleIdentificationDateVehicleFirstRegisteredPropertyCb> m_dateVehicleFirstRegisteredCallbacks;
    // Mutex for m_dateVehicleFirstRegisteredCallbacks
    mutable std::shared_timed_mutex m_dateVehicleFirstRegisteredCallbacksMutex;
    // Next free unique identifier to subscribe for the LicensePlate change.
    std::atomic<long> m_licensePlateChangedCallbackNextId {0};
    // Subscribed callbacks for the LicensePlate change.
    std::map<long, VehicleVehicleIdentificationLicensePlatePropertyCb> m_licensePlateCallbacks;
    // Mutex for m_licensePlateCallbacks
    mutable std::shared_timed_mutex m_licensePlateCallbacksMutex;
    // Next free unique identifier to subscribe for the MeetsEmissionStandard change.
    std::atomic<long> m_meetsEmissionStandardChangedCallbackNextId {0};
    // Subscribed callbacks for the MeetsEmissionStandard change.
    std::map<long, VehicleVehicleIdentificationMeetsEmissionStandardPropertyCb> m_meetsEmissionStandardCallbacks;
    // Mutex for m_meetsEmissionStandardCallbacks
    mutable std::shared_timed_mutex m_meetsEmissionStandardCallbacksMutex;
    // Next free unique identifier to subscribe for the ProductionDate change.
    std::atomic<long> m_productionDateChangedCallbackNextId {0};
    // Subscribed callbacks for the ProductionDate change.
    std::map<long, VehicleVehicleIdentificationProductionDatePropertyCb> m_productionDateCallbacks;
    // Mutex for m_productionDateCallbacks
    mutable std::shared_timed_mutex m_productionDateCallbacksMutex;
    // Next free unique identifier to subscribe for the PurchaseDate change.
    std::atomic<long> m_purchaseDateChangedCallbackNextId {0};
    // Subscribed callbacks for the PurchaseDate change.
    std::map<long, VehicleVehicleIdentificationPurchaseDatePropertyCb> m_purchaseDateCallbacks;
    // Mutex for m_purchaseDateCallbacks
    mutable std::shared_timed_mutex m_purchaseDateCallbacksMutex;
    // Next free unique identifier to subscribe for the VehicleModelDate change.
    std::atomic<long> m_vehicleModelDateChangedCallbackNextId {0};
    // Subscribed callbacks for the VehicleModelDate change.
    std::map<long, VehicleVehicleIdentificationVehicleModelDatePropertyCb> m_vehicleModelDateCallbacks;
    // Mutex for m_vehicleModelDateCallbacks
    mutable std::shared_timed_mutex m_vehicleModelDateCallbacksMutex;
    // Next free unique identifier to subscribe for the VehicleConfiguration change.
    std::atomic<long> m_vehicleConfigurationChangedCallbackNextId {0};
    // Subscribed callbacks for the VehicleConfiguration change.
    std::map<long, VehicleVehicleIdentificationVehicleConfigurationPropertyCb> m_vehicleConfigurationCallbacks;
    // Mutex for m_vehicleConfigurationCallbacks
    mutable std::shared_timed_mutex m_vehicleConfigurationCallbacksMutex;
    // Next free unique identifier to subscribe for the VehicleSeatingCapacity change.
    std::atomic<long> m_vehicleSeatingCapacityChangedCallbackNextId {0};
    // Subscribed callbacks for the VehicleSeatingCapacity change.
    std::map<long, VehicleVehicleIdentificationVehicleSeatingCapacityPropertyCb> m_vehicleSeatingCapacityCallbacks;
    // Mutex for m_vehicleSeatingCapacityCallbacks
    mutable std::shared_timed_mutex m_vehicleSeatingCapacityCallbacksMutex;
    // Next free unique identifier to subscribe for the VehicleSpecialUsage change.
    std::atomic<long> m_vehicleSpecialUsageChangedCallbackNextId {0};
    // Subscribed callbacks for the VehicleSpecialUsage change.
    std::map<long, VehicleVehicleIdentificationVehicleSpecialUsagePropertyCb> m_vehicleSpecialUsageCallbacks;
    // Mutex for m_vehicleSpecialUsageCallbacks
    mutable std::shared_timed_mutex m_vehicleSpecialUsageCallbacksMutex;
    // Next free unique identifier to subscribe for the VehicleExteriorColor change.
    std::atomic<long> m_vehicleExteriorColorChangedCallbackNextId {0};
    // Subscribed callbacks for the VehicleExteriorColor change.
    std::map<long, VehicleVehicleIdentificationVehicleExteriorColorPropertyCb> m_vehicleExteriorColorCallbacks;
    // Mutex for m_vehicleExteriorColorCallbacks
    mutable std::shared_timed_mutex m_vehicleExteriorColorCallbacksMutex;
    // Next free unique identifier to subscribe for the VehicleInteriorColor change.
    std::atomic<long> m_vehicleInteriorColorChangedCallbackNextId {0};
    // Subscribed callbacks for the VehicleInteriorColor change.
    std::map<long, VehicleVehicleIdentificationVehicleInteriorColorPropertyCb> m_vehicleInteriorColorCallbacks;
    // Mutex for m_vehicleInteriorColorCallbacks
    mutable std::shared_timed_mutex m_vehicleInteriorColorCallbacksMutex;
    // Next free unique identifier to subscribe for the VehicleInteriorType change.
    std::atomic<long> m_vehicleInteriorTypeChangedCallbackNextId {0};
    // Subscribed callbacks for the VehicleInteriorType change.
    std::map<long, VehicleVehicleIdentificationVehicleInteriorTypePropertyCb> m_vehicleInteriorTypeCallbacks;
    // Mutex for m_vehicleInteriorTypeCallbacks
    mutable std::shared_timed_mutex m_vehicleInteriorTypeCallbacksMutex;
    // Next free unique identifier to subscribe for the KnownVehicleDamages change.
    std::atomic<long> m_knownVehicleDamagesChangedCallbackNextId {0};
    // Subscribed callbacks for the KnownVehicleDamages change.
    std::map<long, VehicleVehicleIdentificationKnownVehicleDamagesPropertyCb> m_knownVehicleDamagesCallbacks;
    // Mutex for m_knownVehicleDamagesCallbacks
    mutable std::shared_timed_mutex m_knownVehicleDamagesCallbacksMutex;
    // Next free unique identifier to subscribe for the OptionalExtras change.
    std::atomic<long> m_optionalExtrasChangedCallbackNextId {0};
    // Subscribed callbacks for the OptionalExtras change.
    std::map<long, VehicleVehicleIdentificationOptionalExtrasPropertyCb> m_optionalExtrasCallbacks;
    // Mutex for m_optionalExtrasCallbacks
    mutable std::shared_timed_mutex m_optionalExtrasCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
