#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehicleVehicleIdentificationSubscriber;
class IVehicleVehicleIdentificationPublisher;

/**
*
* IVehicleVehicleIdentification provides an interface for
 *  - methods defined for your Vehicle_VehicleIdentification 
 *  - property setters and getters for defined properties
 * The IVehicleVehicleIdentification also provides an interface to access a publisher IVehicleVehicleIdentificationPublisher, a class used by IVehicleVehicleIdentificationSubscriber clients.
 * The implementation should notify the publisher IVehicleVehicleIdentificationPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehicleVehicleIdentificationSubscriber, IVehicleVehicleIdentificationPublisher
 * and the example implementation Vehicle_VehicleIdentification  or the
 */
class CPP_VEHICLE_EXPORT IVehicleVehicleIdentification
{
public:
    virtual ~IVehicleVehicleIdentification() = default;

    /**
    * Sets the value of the VIN property.
    * @param VIN 17-character Vehicle Identification Number (VIN) as defined by ISO 3779.
    */
    virtual void setVin(const std::string& VIN) = 0;
    /**
    * Gets the value of the VIN property.
    * @return 17-character Vehicle Identification Number (VIN) as defined by ISO 3779.
    */
    virtual const std::string& getVin() const = 0;

    /**
    * Sets the value of the WMI property.
    * @param WMI 3-character World Manufacturer Identification (WMI) as defined by ISO 3780.
    */
    virtual void setWmi(const std::string& WMI) = 0;
    /**
    * Gets the value of the WMI property.
    * @return 3-character World Manufacturer Identification (WMI) as defined by ISO 3780.
    */
    virtual const std::string& getWmi() const = 0;

    /**
    * Sets the value of the Brand property.
    * @param Brand Vehicle brand or manufacturer.
    */
    virtual void setBrand(const std::string& Brand) = 0;
    /**
    * Gets the value of the Brand property.
    * @return Vehicle brand or manufacturer.
    */
    virtual const std::string& getBrand() const = 0;

    /**
    * Sets the value of the Model property.
    * @param Model Vehicle model.
    */
    virtual void setModel(const std::string& Model) = 0;
    /**
    * Gets the value of the Model property.
    * @return Vehicle model.
    */
    virtual const std::string& getModel() const = 0;

    /**
    * Sets the value of the Year property.
    * @param Year Model year of the vehicle.
    */
    virtual void setYear(int32_t Year) = 0;
    /**
    * Gets the value of the Year property.
    * @return Model year of the vehicle.
    */
    virtual int32_t getYear() const = 0;

    /**
    * Sets the value of the AcrissCode property.
    * @param AcrissCode The ACRISS Car Classification Code is a code used by many car rental companies.
    */
    virtual void setAcrissCode(const std::string& AcrissCode) = 0;
    /**
    * Gets the value of the AcrissCode property.
    * @return The ACRISS Car Classification Code is a code used by many car rental companies.
    */
    virtual const std::string& getAcrissCode() const = 0;

    /**
    * Sets the value of the BodyType property.
    * @param BodyType Indicates the design and body style of the vehicle (e.g. station wagon, hatchback, etc.).
    */
    virtual void setBodyType(const std::string& BodyType) = 0;
    /**
    * Gets the value of the BodyType property.
    * @return Indicates the design and body style of the vehicle (e.g. station wagon, hatchback, etc.).
    */
    virtual const std::string& getBodyType() const = 0;

    /**
    * Sets the value of the DateVehicleFirstRegistered property.
    * @param DateVehicleFirstRegistered The date in ISO 8601 format of the first registration of the vehicle with the respective public authorities.
    */
    virtual void setDateVehicleFirstRegistered(const std::string& DateVehicleFirstRegistered) = 0;
    /**
    * Gets the value of the DateVehicleFirstRegistered property.
    * @return The date in ISO 8601 format of the first registration of the vehicle with the respective public authorities.
    */
    virtual const std::string& getDateVehicleFirstRegistered() const = 0;

    /**
    * Sets the value of the LicensePlate property.
    * @param LicensePlate The license plate of the vehicle.
    */
    virtual void setLicensePlate(const std::string& LicensePlate) = 0;
    /**
    * Gets the value of the LicensePlate property.
    * @return The license plate of the vehicle.
    */
    virtual const std::string& getLicensePlate() const = 0;

    /**
    * Sets the value of the MeetsEmissionStandard property.
    * @param MeetsEmissionStandard Indicates that the vehicle meets the respective emission standard.
    */
    virtual void setMeetsEmissionStandard(const std::string& MeetsEmissionStandard) = 0;
    /**
    * Gets the value of the MeetsEmissionStandard property.
    * @return Indicates that the vehicle meets the respective emission standard.
    */
    virtual const std::string& getMeetsEmissionStandard() const = 0;

    /**
    * Sets the value of the ProductionDate property.
    * @param ProductionDate The date in ISO 8601 format of production of the item, e.g. vehicle.
    */
    virtual void setProductionDate(const std::string& ProductionDate) = 0;
    /**
    * Gets the value of the ProductionDate property.
    * @return The date in ISO 8601 format of production of the item, e.g. vehicle.
    */
    virtual const std::string& getProductionDate() const = 0;

    /**
    * Sets the value of the PurchaseDate property.
    * @param PurchaseDate The date in ISO 8601 format of the item e.g. vehicle was purchased by the current owner.
    */
    virtual void setPurchaseDate(const std::string& PurchaseDate) = 0;
    /**
    * Gets the value of the PurchaseDate property.
    * @return The date in ISO 8601 format of the item e.g. vehicle was purchased by the current owner.
    */
    virtual const std::string& getPurchaseDate() const = 0;

    /**
    * Sets the value of the VehicleModelDate property.
    * @param VehicleModelDate The release date in ISO 8601 format of a vehicle model (often used to differentiate versions of the same make and model).
    */
    virtual void setVehicleModelDate(const std::string& VehicleModelDate) = 0;
    /**
    * Gets the value of the VehicleModelDate property.
    * @return The release date in ISO 8601 format of a vehicle model (often used to differentiate versions of the same make and model).
    */
    virtual const std::string& getVehicleModelDate() const = 0;

    /**
    * Sets the value of the VehicleConfiguration property.
    * @param VehicleConfiguration A short text indicating the configuration of the vehicle, e.g. '5dr hatchback ST 2.5 MT 225 hp' or 'limited edition'.
    */
    virtual void setVehicleConfiguration(const std::string& VehicleConfiguration) = 0;
    /**
    * Gets the value of the VehicleConfiguration property.
    * @return A short text indicating the configuration of the vehicle, e.g. '5dr hatchback ST 2.5 MT 225 hp' or 'limited edition'.
    */
    virtual const std::string& getVehicleConfiguration() const = 0;

    /**
    * Sets the value of the VehicleSeatingCapacity property.
    * @param VehicleSeatingCapacity The number of passengers that can be seated in the vehicle, both in terms of the physical space available, and in terms of limitations set by law.
    */
    virtual void setVehicleSeatingCapacity(int32_t VehicleSeatingCapacity) = 0;
    /**
    * Gets the value of the VehicleSeatingCapacity property.
    * @return The number of passengers that can be seated in the vehicle, both in terms of the physical space available, and in terms of limitations set by law.
    */
    virtual int32_t getVehicleSeatingCapacity() const = 0;

    /**
    * Sets the value of the VehicleSpecialUsage property.
    * @param VehicleSpecialUsage Indicates whether the vehicle has been used for special purposes, like commercial rental, driving school.
    */
    virtual void setVehicleSpecialUsage(const std::string& VehicleSpecialUsage) = 0;
    /**
    * Gets the value of the VehicleSpecialUsage property.
    * @return Indicates whether the vehicle has been used for special purposes, like commercial rental, driving school.
    */
    virtual const std::string& getVehicleSpecialUsage() const = 0;

    /**
    * Sets the value of the VehicleExteriorColor property.
    * @param VehicleExteriorColor The main color of the exterior within the basic color palette (eg. red, blue, black, white, ...).
    */
    virtual void setVehicleExteriorColor(const std::string& VehicleExteriorColor) = 0;
    /**
    * Gets the value of the VehicleExteriorColor property.
    * @return The main color of the exterior within the basic color palette (eg. red, blue, black, white, ...).
    */
    virtual const std::string& getVehicleExteriorColor() const = 0;

    /**
    * Sets the value of the VehicleInteriorColor property.
    * @param VehicleInteriorColor The color or color combination of the interior of the vehicle.
    */
    virtual void setVehicleInteriorColor(const std::string& VehicleInteriorColor) = 0;
    /**
    * Gets the value of the VehicleInteriorColor property.
    * @return The color or color combination of the interior of the vehicle.
    */
    virtual const std::string& getVehicleInteriorColor() const = 0;

    /**
    * Sets the value of the VehicleInteriorType property.
    * @param VehicleInteriorType The type or material of the interior of the vehicle (e.g. synthetic fabric, leather, wood, etc.).
    */
    virtual void setVehicleInteriorType(const std::string& VehicleInteriorType) = 0;
    /**
    * Gets the value of the VehicleInteriorType property.
    * @return The type or material of the interior of the vehicle (e.g. synthetic fabric, leather, wood, etc.).
    */
    virtual const std::string& getVehicleInteriorType() const = 0;

    /**
    * Sets the value of the KnownVehicleDamages property.
    * @param KnownVehicleDamages A textual description of known damages, both repaired and unrepaired.
    */
    virtual void setKnownVehicleDamages(const std::string& KnownVehicleDamages) = 0;
    /**
    * Gets the value of the KnownVehicleDamages property.
    * @return A textual description of known damages, both repaired and unrepaired.
    */
    virtual const std::string& getKnownVehicleDamages() const = 0;

    /**
    * Sets the value of the OptionalExtras property.
    * @param OptionalExtras Optional extras refers to all car equipment options that are not installed as standard by the manufacturer.
    */
    virtual void setOptionalExtras(const std::list<std::string>& OptionalExtras) = 0;
    /**
    * Gets the value of the OptionalExtras property.
    * @return Optional extras refers to all car equipment options that are not installed as standard by the manufacturer.
    */
    virtual const std::list<std::string>& getOptionalExtras() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_VehicleIdentification changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_VehicleIdentification.
    */
    virtual IVehicleVehicleIdentificationPublisher& _getPublisher() const = 0;
};


/**
 * The IVehicleVehicleIdentificationSubscriber contains functions to allow informing about signals or property changes of the IVehicleVehicleIdentification implementation.
 * The implementation for IVehicleVehicleIdentification should provide mechanism for subscription of the IVehicleVehicleIdentificationSubscriber clients.
 * See IVehicleVehicleIdentificationPublisher, which provides facilitation for this purpose.
 * The implementation for IVehicleVehicleIdentification should call the IVehicleVehicleIdentificationSubscriber interface functions on either signal emit or property change.
 * You can use IVehicleVehicleIdentificationSubscriber class to implement clients of the IVehicleVehicleIdentification or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehicleVehicleIdentificationSubscriber
{
public:
    virtual ~IVehicleVehicleIdentificationSubscriber() = default;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when VIN value has changed if subscribed for the VIN change.
    * @param VIN 17-character Vehicle Identification Number (VIN) as defined by ISO 3779.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onVinChanged(const std::string& VIN) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when WMI value has changed if subscribed for the WMI change.
    * @param WMI 3-character World Manufacturer Identification (WMI) as defined by ISO 3780.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onWmiChanged(const std::string& WMI) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when Brand value has changed if subscribed for the Brand change.
    * @param Brand Vehicle brand or manufacturer.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onBrandChanged(const std::string& Brand) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when Model value has changed if subscribed for the Model change.
    * @param Model Vehicle model.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onModelChanged(const std::string& Model) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when Year value has changed if subscribed for the Year change.
    * @param Year Model year of the vehicle.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onYearChanged(int32_t Year) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when AcrissCode value has changed if subscribed for the AcrissCode change.
    * @param AcrissCode The ACRISS Car Classification Code is a code used by many car rental companies.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onAcrissCodeChanged(const std::string& AcrissCode) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when BodyType value has changed if subscribed for the BodyType change.
    * @param BodyType Indicates the design and body style of the vehicle (e.g. station wagon, hatchback, etc.).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onBodyTypeChanged(const std::string& BodyType) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when DateVehicleFirstRegistered value has changed if subscribed for the DateVehicleFirstRegistered change.
    * @param DateVehicleFirstRegistered The date in ISO 8601 format of the first registration of the vehicle with the respective public authorities.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onDateVehicleFirstRegisteredChanged(const std::string& DateVehicleFirstRegistered) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when LicensePlate value has changed if subscribed for the LicensePlate change.
    * @param LicensePlate The license plate of the vehicle.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onLicensePlateChanged(const std::string& LicensePlate) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when MeetsEmissionStandard value has changed if subscribed for the MeetsEmissionStandard change.
    * @param MeetsEmissionStandard Indicates that the vehicle meets the respective emission standard.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onMeetsEmissionStandardChanged(const std::string& MeetsEmissionStandard) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when ProductionDate value has changed if subscribed for the ProductionDate change.
    * @param ProductionDate The date in ISO 8601 format of production of the item, e.g. vehicle.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onProductionDateChanged(const std::string& ProductionDate) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when PurchaseDate value has changed if subscribed for the PurchaseDate change.
    * @param PurchaseDate The date in ISO 8601 format of the item e.g. vehicle was purchased by the current owner.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPurchaseDateChanged(const std::string& PurchaseDate) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when VehicleModelDate value has changed if subscribed for the VehicleModelDate change.
    * @param VehicleModelDate The release date in ISO 8601 format of a vehicle model (often used to differentiate versions of the same make and model).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onVehicleModelDateChanged(const std::string& VehicleModelDate) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when VehicleConfiguration value has changed if subscribed for the VehicleConfiguration change.
    * @param VehicleConfiguration A short text indicating the configuration of the vehicle, e.g. '5dr hatchback ST 2.5 MT 225 hp' or 'limited edition'.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onVehicleConfigurationChanged(const std::string& VehicleConfiguration) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when VehicleSeatingCapacity value has changed if subscribed for the VehicleSeatingCapacity change.
    * @param VehicleSeatingCapacity The number of passengers that can be seated in the vehicle, both in terms of the physical space available, and in terms of limitations set by law.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onVehicleSeatingCapacityChanged(int32_t VehicleSeatingCapacity) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when VehicleSpecialUsage value has changed if subscribed for the VehicleSpecialUsage change.
    * @param VehicleSpecialUsage Indicates whether the vehicle has been used for special purposes, like commercial rental, driving school.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onVehicleSpecialUsageChanged(const std::string& VehicleSpecialUsage) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when VehicleExteriorColor value has changed if subscribed for the VehicleExteriorColor change.
    * @param VehicleExteriorColor The main color of the exterior within the basic color palette (eg. red, blue, black, white, ...).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onVehicleExteriorColorChanged(const std::string& VehicleExteriorColor) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when VehicleInteriorColor value has changed if subscribed for the VehicleInteriorColor change.
    * @param VehicleInteriorColor The color or color combination of the interior of the vehicle.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onVehicleInteriorColorChanged(const std::string& VehicleInteriorColor) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when VehicleInteriorType value has changed if subscribed for the VehicleInteriorType change.
    * @param VehicleInteriorType The type or material of the interior of the vehicle (e.g. synthetic fabric, leather, wood, etc.).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onVehicleInteriorTypeChanged(const std::string& VehicleInteriorType) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when KnownVehicleDamages value has changed if subscribed for the KnownVehicleDamages change.
    * @param KnownVehicleDamages A textual description of known damages, both repaired and unrepaired.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onKnownVehicleDamagesChanged(const std::string& KnownVehicleDamages) = 0;
    /**
    * Called by the IVehicleVehicleIdentificationPublisher when OptionalExtras value has changed if subscribed for the OptionalExtras change.
    * @param OptionalExtras Optional extras refers to all car equipment options that are not installed as standard by the manufacturer.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onOptionalExtrasChanged(const std::list<std::string>& OptionalExtras) = 0;
};

/** Callback for changes of VIN */
using VehicleVehicleIdentificationVinPropertyCb = std::function<void(const std::string& VIN)>;
/** Callback for changes of WMI */
using VehicleVehicleIdentificationWmiPropertyCb = std::function<void(const std::string& WMI)>;
/** Callback for changes of Brand */
using VehicleVehicleIdentificationBrandPropertyCb = std::function<void(const std::string& Brand)>;
/** Callback for changes of Model */
using VehicleVehicleIdentificationModelPropertyCb = std::function<void(const std::string& Model)>;
/** Callback for changes of Year */
using VehicleVehicleIdentificationYearPropertyCb = std::function<void(int32_t Year)>;
/** Callback for changes of AcrissCode */
using VehicleVehicleIdentificationAcrissCodePropertyCb = std::function<void(const std::string& AcrissCode)>;
/** Callback for changes of BodyType */
using VehicleVehicleIdentificationBodyTypePropertyCb = std::function<void(const std::string& BodyType)>;
/** Callback for changes of DateVehicleFirstRegistered */
using VehicleVehicleIdentificationDateVehicleFirstRegisteredPropertyCb = std::function<void(const std::string& DateVehicleFirstRegistered)>;
/** Callback for changes of LicensePlate */
using VehicleVehicleIdentificationLicensePlatePropertyCb = std::function<void(const std::string& LicensePlate)>;
/** Callback for changes of MeetsEmissionStandard */
using VehicleVehicleIdentificationMeetsEmissionStandardPropertyCb = std::function<void(const std::string& MeetsEmissionStandard)>;
/** Callback for changes of ProductionDate */
using VehicleVehicleIdentificationProductionDatePropertyCb = std::function<void(const std::string& ProductionDate)>;
/** Callback for changes of PurchaseDate */
using VehicleVehicleIdentificationPurchaseDatePropertyCb = std::function<void(const std::string& PurchaseDate)>;
/** Callback for changes of VehicleModelDate */
using VehicleVehicleIdentificationVehicleModelDatePropertyCb = std::function<void(const std::string& VehicleModelDate)>;
/** Callback for changes of VehicleConfiguration */
using VehicleVehicleIdentificationVehicleConfigurationPropertyCb = std::function<void(const std::string& VehicleConfiguration)>;
/** Callback for changes of VehicleSeatingCapacity */
using VehicleVehicleIdentificationVehicleSeatingCapacityPropertyCb = std::function<void(int32_t VehicleSeatingCapacity)>;
/** Callback for changes of VehicleSpecialUsage */
using VehicleVehicleIdentificationVehicleSpecialUsagePropertyCb = std::function<void(const std::string& VehicleSpecialUsage)>;
/** Callback for changes of VehicleExteriorColor */
using VehicleVehicleIdentificationVehicleExteriorColorPropertyCb = std::function<void(const std::string& VehicleExteriorColor)>;
/** Callback for changes of VehicleInteriorColor */
using VehicleVehicleIdentificationVehicleInteriorColorPropertyCb = std::function<void(const std::string& VehicleInteriorColor)>;
/** Callback for changes of VehicleInteriorType */
using VehicleVehicleIdentificationVehicleInteriorTypePropertyCb = std::function<void(const std::string& VehicleInteriorType)>;
/** Callback for changes of KnownVehicleDamages */
using VehicleVehicleIdentificationKnownVehicleDamagesPropertyCb = std::function<void(const std::string& KnownVehicleDamages)>;
/** Callback for changes of OptionalExtras */
using VehicleVehicleIdentificationOptionalExtrasPropertyCb = std::function<void(const std::list<std::string>& OptionalExtras)>;


/**
 * The IVehicleVehicleIdentificationPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehicleVehicleIdentification implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehicleVehicleIdentificationSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehicleVehicleIdentification on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehicleVehicleIdentificationPublisher
{
public:
    virtual ~IVehicleVehicleIdentificationPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_VehicleIdentification.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehicleVehicleIdentificationSubscriber which is subscribed in this function to any change of the Vehicle_VehicleIdentification.
    */
    virtual void subscribeToAllChanges(IVehicleVehicleIdentificationSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_VehicleIdentification.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehicleVehicleIdentificationSubscriber which subscription for any change of the Vehicle_VehicleIdentification is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehicleVehicleIdentificationSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for VIN value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationVinPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToVinChanged(VehicleVehicleIdentificationVinPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from VIN property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromVinChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for WMI value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationWmiPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToWmiChanged(VehicleVehicleIdentificationWmiPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from WMI property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromWmiChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Brand value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationBrandPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToBrandChanged(VehicleVehicleIdentificationBrandPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Brand property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromBrandChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Model value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationModelPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToModelChanged(VehicleVehicleIdentificationModelPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Model property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromModelChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Year value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationYearPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToYearChanged(VehicleVehicleIdentificationYearPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Year property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromYearChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for AcrissCode value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationAcrissCodePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToAcrissCodeChanged(VehicleVehicleIdentificationAcrissCodePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from AcrissCode property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromAcrissCodeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for BodyType value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationBodyTypePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToBodyTypeChanged(VehicleVehicleIdentificationBodyTypePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from BodyType property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromBodyTypeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for DateVehicleFirstRegistered value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationDateVehicleFirstRegisteredPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToDateVehicleFirstRegisteredChanged(VehicleVehicleIdentificationDateVehicleFirstRegisteredPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from DateVehicleFirstRegistered property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromDateVehicleFirstRegisteredChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for LicensePlate value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationLicensePlatePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToLicensePlateChanged(VehicleVehicleIdentificationLicensePlatePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from LicensePlate property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromLicensePlateChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for MeetsEmissionStandard value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationMeetsEmissionStandardPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToMeetsEmissionStandardChanged(VehicleVehicleIdentificationMeetsEmissionStandardPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from MeetsEmissionStandard property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromMeetsEmissionStandardChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for ProductionDate value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationProductionDatePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToProductionDateChanged(VehicleVehicleIdentificationProductionDatePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from ProductionDate property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromProductionDateChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for PurchaseDate value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationPurchaseDatePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPurchaseDateChanged(VehicleVehicleIdentificationPurchaseDatePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from PurchaseDate property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPurchaseDateChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for VehicleModelDate value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationVehicleModelDatePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToVehicleModelDateChanged(VehicleVehicleIdentificationVehicleModelDatePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from VehicleModelDate property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromVehicleModelDateChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for VehicleConfiguration value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationVehicleConfigurationPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToVehicleConfigurationChanged(VehicleVehicleIdentificationVehicleConfigurationPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from VehicleConfiguration property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromVehicleConfigurationChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for VehicleSeatingCapacity value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationVehicleSeatingCapacityPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToVehicleSeatingCapacityChanged(VehicleVehicleIdentificationVehicleSeatingCapacityPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from VehicleSeatingCapacity property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromVehicleSeatingCapacityChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for VehicleSpecialUsage value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationVehicleSpecialUsagePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToVehicleSpecialUsageChanged(VehicleVehicleIdentificationVehicleSpecialUsagePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from VehicleSpecialUsage property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromVehicleSpecialUsageChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for VehicleExteriorColor value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationVehicleExteriorColorPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToVehicleExteriorColorChanged(VehicleVehicleIdentificationVehicleExteriorColorPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from VehicleExteriorColor property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromVehicleExteriorColorChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for VehicleInteriorColor value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationVehicleInteriorColorPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToVehicleInteriorColorChanged(VehicleVehicleIdentificationVehicleInteriorColorPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from VehicleInteriorColor property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromVehicleInteriorColorChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for VehicleInteriorType value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationVehicleInteriorTypePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToVehicleInteriorTypeChanged(VehicleVehicleIdentificationVehicleInteriorTypePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from VehicleInteriorType property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromVehicleInteriorTypeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for KnownVehicleDamages value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationKnownVehicleDamagesPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToKnownVehicleDamagesChanged(VehicleVehicleIdentificationKnownVehicleDamagesPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from KnownVehicleDamages property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromKnownVehicleDamagesChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for OptionalExtras value changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehicleVehicleIdentificationOptionalExtrasPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToOptionalExtrasChanged(VehicleVehicleIdentificationOptionalExtrasPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from OptionalExtras property changes.
    * If your subscriber uses subscription with IVehicleVehicleIdentificationSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromOptionalExtrasChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property VIN changes.
    * @param The new value of VIN.
    * 17-character Vehicle Identification Number (VIN) as defined by ISO 3779.
    */
    virtual void publishVinChanged(const std::string& VIN) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property WMI changes.
    * @param The new value of WMI.
    * 3-character World Manufacturer Identification (WMI) as defined by ISO 3780.
    */
    virtual void publishWmiChanged(const std::string& WMI) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property Brand changes.
    * @param The new value of Brand.
    * Vehicle brand or manufacturer.
    */
    virtual void publishBrandChanged(const std::string& Brand) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property Model changes.
    * @param The new value of Model.
    * Vehicle model.
    */
    virtual void publishModelChanged(const std::string& Model) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property Year changes.
    * @param The new value of Year.
    * Model year of the vehicle.
    */
    virtual void publishYearChanged(int32_t Year) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property AcrissCode changes.
    * @param The new value of AcrissCode.
    * The ACRISS Car Classification Code is a code used by many car rental companies.
    */
    virtual void publishAcrissCodeChanged(const std::string& AcrissCode) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property BodyType changes.
    * @param The new value of BodyType.
    * Indicates the design and body style of the vehicle (e.g. station wagon, hatchback, etc.).
    */
    virtual void publishBodyTypeChanged(const std::string& BodyType) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property DateVehicleFirstRegistered changes.
    * @param The new value of DateVehicleFirstRegistered.
    * The date in ISO 8601 format of the first registration of the vehicle with the respective public authorities.
    */
    virtual void publishDateVehicleFirstRegisteredChanged(const std::string& DateVehicleFirstRegistered) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property LicensePlate changes.
    * @param The new value of LicensePlate.
    * The license plate of the vehicle.
    */
    virtual void publishLicensePlateChanged(const std::string& LicensePlate) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property MeetsEmissionStandard changes.
    * @param The new value of MeetsEmissionStandard.
    * Indicates that the vehicle meets the respective emission standard.
    */
    virtual void publishMeetsEmissionStandardChanged(const std::string& MeetsEmissionStandard) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property ProductionDate changes.
    * @param The new value of ProductionDate.
    * The date in ISO 8601 format of production of the item, e.g. vehicle.
    */
    virtual void publishProductionDateChanged(const std::string& ProductionDate) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property PurchaseDate changes.
    * @param The new value of PurchaseDate.
    * The date in ISO 8601 format of the item e.g. vehicle was purchased by the current owner.
    */
    virtual void publishPurchaseDateChanged(const std::string& PurchaseDate) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property VehicleModelDate changes.
    * @param The new value of VehicleModelDate.
    * The release date in ISO 8601 format of a vehicle model (often used to differentiate versions of the same make and model).
    */
    virtual void publishVehicleModelDateChanged(const std::string& VehicleModelDate) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property VehicleConfiguration changes.
    * @param The new value of VehicleConfiguration.
    * A short text indicating the configuration of the vehicle, e.g. '5dr hatchback ST 2.5 MT 225 hp' or 'limited edition'.
    */
    virtual void publishVehicleConfigurationChanged(const std::string& VehicleConfiguration) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property VehicleSeatingCapacity changes.
    * @param The new value of VehicleSeatingCapacity.
    * The number of passengers that can be seated in the vehicle, both in terms of the physical space available, and in terms of limitations set by law.
    */
    virtual void publishVehicleSeatingCapacityChanged(int32_t VehicleSeatingCapacity) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property VehicleSpecialUsage changes.
    * @param The new value of VehicleSpecialUsage.
    * Indicates whether the vehicle has been used for special purposes, like commercial rental, driving school.
    */
    virtual void publishVehicleSpecialUsageChanged(const std::string& VehicleSpecialUsage) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property VehicleExteriorColor changes.
    * @param The new value of VehicleExteriorColor.
    * The main color of the exterior within the basic color palette (eg. red, blue, black, white, ...).
    */
    virtual void publishVehicleExteriorColorChanged(const std::string& VehicleExteriorColor) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property VehicleInteriorColor changes.
    * @param The new value of VehicleInteriorColor.
    * The color or color combination of the interior of the vehicle.
    */
    virtual void publishVehicleInteriorColorChanged(const std::string& VehicleInteriorColor) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property VehicleInteriorType changes.
    * @param The new value of VehicleInteriorType.
    * The type or material of the interior of the vehicle (e.g. synthetic fabric, leather, wood, etc.).
    */
    virtual void publishVehicleInteriorTypeChanged(const std::string& VehicleInteriorType) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property KnownVehicleDamages changes.
    * @param The new value of KnownVehicleDamages.
    * A textual description of known damages, both repaired and unrepaired.
    */
    virtual void publishKnownVehicleDamagesChanged(const std::string& KnownVehicleDamages) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_VehicleIdentification implementation when property OptionalExtras changes.
    * @param The new value of OptionalExtras.
    * Optional extras refers to all car equipment options that are not installed as standard by the manufacturer.
    */
    virtual void publishOptionalExtrasChanged(const std::list<std::string>& OptionalExtras) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
