

#include "vehicle/generated/core/vehiclevehicleidentification.publisher.h"
#include <algorithm>


using namespace Cpp::Vehicle;

void VehicleVehicleIdentificationPublisher::subscribeToAllChanges(IVehicleVehicleIdentificationSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found == m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.push_back(std::reference_wrapper<IVehicleVehicleIdentificationSubscriber>(subscriber));
    }
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromAllChanges(IVehicleVehicleIdentificationSubscriber& subscriber)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_allChangesSubscribersMutex);
    auto found = std::find_if(m_allChangesSubscribers.begin(), m_allChangesSubscribers.end(),
                        [&subscriber](const auto element){return &(element.get()) == &subscriber;});
    if (found != m_allChangesSubscribers.end())
    {
        m_allChangesSubscribers.erase(found);
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToVinChanged(VehicleVehicleIdentificationVinPropertyCb callback)
{
    auto handleId = m_vinChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_vinCallbacksMutex);
    m_vinCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromVinChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vinCallbacksMutex);
    m_vinCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishVinChanged(const std::string& VIN) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onVinChanged(VIN);
    }
    std::shared_lock<std::shared_timed_mutex> vinCallbacksLock(m_vinCallbacksMutex);
    const auto vinCallbacks = m_vinCallbacks;
    vinCallbacksLock.unlock();
    for(const auto& callbackEntry: vinCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(VIN);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToWmiChanged(VehicleVehicleIdentificationWmiPropertyCb callback)
{
    auto handleId = m_wmiChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_wmiCallbacksMutex);
    m_wmiCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromWmiChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_wmiCallbacksMutex);
    m_wmiCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishWmiChanged(const std::string& WMI) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onWmiChanged(WMI);
    }
    std::shared_lock<std::shared_timed_mutex> wmiCallbacksLock(m_wmiCallbacksMutex);
    const auto wmiCallbacks = m_wmiCallbacks;
    wmiCallbacksLock.unlock();
    for(const auto& callbackEntry: wmiCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(WMI);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToBrandChanged(VehicleVehicleIdentificationBrandPropertyCb callback)
{
    auto handleId = m_brandChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_brandCallbacksMutex);
    m_brandCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromBrandChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_brandCallbacksMutex);
    m_brandCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishBrandChanged(const std::string& Brand) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onBrandChanged(Brand);
    }
    std::shared_lock<std::shared_timed_mutex> brandCallbacksLock(m_brandCallbacksMutex);
    const auto brandCallbacks = m_brandCallbacks;
    brandCallbacksLock.unlock();
    for(const auto& callbackEntry: brandCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Brand);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToModelChanged(VehicleVehicleIdentificationModelPropertyCb callback)
{
    auto handleId = m_modelChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_modelCallbacksMutex);
    m_modelCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromModelChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_modelCallbacksMutex);
    m_modelCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishModelChanged(const std::string& Model) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onModelChanged(Model);
    }
    std::shared_lock<std::shared_timed_mutex> modelCallbacksLock(m_modelCallbacksMutex);
    const auto modelCallbacks = m_modelCallbacks;
    modelCallbacksLock.unlock();
    for(const auto& callbackEntry: modelCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Model);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToYearChanged(VehicleVehicleIdentificationYearPropertyCb callback)
{
    auto handleId = m_yearChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_yearCallbacksMutex);
    m_yearCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromYearChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_yearCallbacksMutex);
    m_yearCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishYearChanged(int32_t Year) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onYearChanged(Year);
    }
    std::shared_lock<std::shared_timed_mutex> yearCallbacksLock(m_yearCallbacksMutex);
    const auto yearCallbacks = m_yearCallbacks;
    yearCallbacksLock.unlock();
    for(const auto& callbackEntry: yearCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(Year);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToAcrissCodeChanged(VehicleVehicleIdentificationAcrissCodePropertyCb callback)
{
    auto handleId = m_acrissCodeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_acrissCodeCallbacksMutex);
    m_acrissCodeCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromAcrissCodeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_acrissCodeCallbacksMutex);
    m_acrissCodeCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishAcrissCodeChanged(const std::string& AcrissCode) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onAcrissCodeChanged(AcrissCode);
    }
    std::shared_lock<std::shared_timed_mutex> acrissCodeCallbacksLock(m_acrissCodeCallbacksMutex);
    const auto acrissCodeCallbacks = m_acrissCodeCallbacks;
    acrissCodeCallbacksLock.unlock();
    for(const auto& callbackEntry: acrissCodeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(AcrissCode);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToBodyTypeChanged(VehicleVehicleIdentificationBodyTypePropertyCb callback)
{
    auto handleId = m_bodyTypeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_bodyTypeCallbacksMutex);
    m_bodyTypeCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromBodyTypeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_bodyTypeCallbacksMutex);
    m_bodyTypeCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishBodyTypeChanged(const std::string& BodyType) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onBodyTypeChanged(BodyType);
    }
    std::shared_lock<std::shared_timed_mutex> bodyTypeCallbacksLock(m_bodyTypeCallbacksMutex);
    const auto bodyTypeCallbacks = m_bodyTypeCallbacks;
    bodyTypeCallbacksLock.unlock();
    for(const auto& callbackEntry: bodyTypeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(BodyType);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToDateVehicleFirstRegisteredChanged(VehicleVehicleIdentificationDateVehicleFirstRegisteredPropertyCb callback)
{
    auto handleId = m_dateVehicleFirstRegisteredChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_dateVehicleFirstRegisteredCallbacksMutex);
    m_dateVehicleFirstRegisteredCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromDateVehicleFirstRegisteredChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_dateVehicleFirstRegisteredCallbacksMutex);
    m_dateVehicleFirstRegisteredCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishDateVehicleFirstRegisteredChanged(const std::string& DateVehicleFirstRegistered) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onDateVehicleFirstRegisteredChanged(DateVehicleFirstRegistered);
    }
    std::shared_lock<std::shared_timed_mutex> dateVehicleFirstRegisteredCallbacksLock(m_dateVehicleFirstRegisteredCallbacksMutex);
    const auto dateVehicleFirstRegisteredCallbacks = m_dateVehicleFirstRegisteredCallbacks;
    dateVehicleFirstRegisteredCallbacksLock.unlock();
    for(const auto& callbackEntry: dateVehicleFirstRegisteredCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(DateVehicleFirstRegistered);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToLicensePlateChanged(VehicleVehicleIdentificationLicensePlatePropertyCb callback)
{
    auto handleId = m_licensePlateChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_licensePlateCallbacksMutex);
    m_licensePlateCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromLicensePlateChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_licensePlateCallbacksMutex);
    m_licensePlateCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishLicensePlateChanged(const std::string& LicensePlate) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onLicensePlateChanged(LicensePlate);
    }
    std::shared_lock<std::shared_timed_mutex> licensePlateCallbacksLock(m_licensePlateCallbacksMutex);
    const auto licensePlateCallbacks = m_licensePlateCallbacks;
    licensePlateCallbacksLock.unlock();
    for(const auto& callbackEntry: licensePlateCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(LicensePlate);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToMeetsEmissionStandardChanged(VehicleVehicleIdentificationMeetsEmissionStandardPropertyCb callback)
{
    auto handleId = m_meetsEmissionStandardChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_meetsEmissionStandardCallbacksMutex);
    m_meetsEmissionStandardCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromMeetsEmissionStandardChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_meetsEmissionStandardCallbacksMutex);
    m_meetsEmissionStandardCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishMeetsEmissionStandardChanged(const std::string& MeetsEmissionStandard) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onMeetsEmissionStandardChanged(MeetsEmissionStandard);
    }
    std::shared_lock<std::shared_timed_mutex> meetsEmissionStandardCallbacksLock(m_meetsEmissionStandardCallbacksMutex);
    const auto meetsEmissionStandardCallbacks = m_meetsEmissionStandardCallbacks;
    meetsEmissionStandardCallbacksLock.unlock();
    for(const auto& callbackEntry: meetsEmissionStandardCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(MeetsEmissionStandard);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToProductionDateChanged(VehicleVehicleIdentificationProductionDatePropertyCb callback)
{
    auto handleId = m_productionDateChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_productionDateCallbacksMutex);
    m_productionDateCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromProductionDateChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_productionDateCallbacksMutex);
    m_productionDateCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishProductionDateChanged(const std::string& ProductionDate) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onProductionDateChanged(ProductionDate);
    }
    std::shared_lock<std::shared_timed_mutex> productionDateCallbacksLock(m_productionDateCallbacksMutex);
    const auto productionDateCallbacks = m_productionDateCallbacks;
    productionDateCallbacksLock.unlock();
    for(const auto& callbackEntry: productionDateCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(ProductionDate);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToPurchaseDateChanged(VehicleVehicleIdentificationPurchaseDatePropertyCb callback)
{
    auto handleId = m_purchaseDateChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_purchaseDateCallbacksMutex);
    m_purchaseDateCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromPurchaseDateChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_purchaseDateCallbacksMutex);
    m_purchaseDateCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishPurchaseDateChanged(const std::string& PurchaseDate) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onPurchaseDateChanged(PurchaseDate);
    }
    std::shared_lock<std::shared_timed_mutex> purchaseDateCallbacksLock(m_purchaseDateCallbacksMutex);
    const auto purchaseDateCallbacks = m_purchaseDateCallbacks;
    purchaseDateCallbacksLock.unlock();
    for(const auto& callbackEntry: purchaseDateCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(PurchaseDate);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToVehicleModelDateChanged(VehicleVehicleIdentificationVehicleModelDatePropertyCb callback)
{
    auto handleId = m_vehicleModelDateChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleModelDateCallbacksMutex);
    m_vehicleModelDateCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromVehicleModelDateChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleModelDateCallbacksMutex);
    m_vehicleModelDateCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishVehicleModelDateChanged(const std::string& VehicleModelDate) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onVehicleModelDateChanged(VehicleModelDate);
    }
    std::shared_lock<std::shared_timed_mutex> vehicleModelDateCallbacksLock(m_vehicleModelDateCallbacksMutex);
    const auto vehicleModelDateCallbacks = m_vehicleModelDateCallbacks;
    vehicleModelDateCallbacksLock.unlock();
    for(const auto& callbackEntry: vehicleModelDateCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(VehicleModelDate);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToVehicleConfigurationChanged(VehicleVehicleIdentificationVehicleConfigurationPropertyCb callback)
{
    auto handleId = m_vehicleConfigurationChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleConfigurationCallbacksMutex);
    m_vehicleConfigurationCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromVehicleConfigurationChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleConfigurationCallbacksMutex);
    m_vehicleConfigurationCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishVehicleConfigurationChanged(const std::string& VehicleConfiguration) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onVehicleConfigurationChanged(VehicleConfiguration);
    }
    std::shared_lock<std::shared_timed_mutex> vehicleConfigurationCallbacksLock(m_vehicleConfigurationCallbacksMutex);
    const auto vehicleConfigurationCallbacks = m_vehicleConfigurationCallbacks;
    vehicleConfigurationCallbacksLock.unlock();
    for(const auto& callbackEntry: vehicleConfigurationCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(VehicleConfiguration);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToVehicleSeatingCapacityChanged(VehicleVehicleIdentificationVehicleSeatingCapacityPropertyCb callback)
{
    auto handleId = m_vehicleSeatingCapacityChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleSeatingCapacityCallbacksMutex);
    m_vehicleSeatingCapacityCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromVehicleSeatingCapacityChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleSeatingCapacityCallbacksMutex);
    m_vehicleSeatingCapacityCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishVehicleSeatingCapacityChanged(int32_t VehicleSeatingCapacity) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onVehicleSeatingCapacityChanged(VehicleSeatingCapacity);
    }
    std::shared_lock<std::shared_timed_mutex> vehicleSeatingCapacityCallbacksLock(m_vehicleSeatingCapacityCallbacksMutex);
    const auto vehicleSeatingCapacityCallbacks = m_vehicleSeatingCapacityCallbacks;
    vehicleSeatingCapacityCallbacksLock.unlock();
    for(const auto& callbackEntry: vehicleSeatingCapacityCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(VehicleSeatingCapacity);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToVehicleSpecialUsageChanged(VehicleVehicleIdentificationVehicleSpecialUsagePropertyCb callback)
{
    auto handleId = m_vehicleSpecialUsageChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleSpecialUsageCallbacksMutex);
    m_vehicleSpecialUsageCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromVehicleSpecialUsageChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleSpecialUsageCallbacksMutex);
    m_vehicleSpecialUsageCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishVehicleSpecialUsageChanged(const std::string& VehicleSpecialUsage) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onVehicleSpecialUsageChanged(VehicleSpecialUsage);
    }
    std::shared_lock<std::shared_timed_mutex> vehicleSpecialUsageCallbacksLock(m_vehicleSpecialUsageCallbacksMutex);
    const auto vehicleSpecialUsageCallbacks = m_vehicleSpecialUsageCallbacks;
    vehicleSpecialUsageCallbacksLock.unlock();
    for(const auto& callbackEntry: vehicleSpecialUsageCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(VehicleSpecialUsage);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToVehicleExteriorColorChanged(VehicleVehicleIdentificationVehicleExteriorColorPropertyCb callback)
{
    auto handleId = m_vehicleExteriorColorChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleExteriorColorCallbacksMutex);
    m_vehicleExteriorColorCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromVehicleExteriorColorChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleExteriorColorCallbacksMutex);
    m_vehicleExteriorColorCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishVehicleExteriorColorChanged(const std::string& VehicleExteriorColor) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onVehicleExteriorColorChanged(VehicleExteriorColor);
    }
    std::shared_lock<std::shared_timed_mutex> vehicleExteriorColorCallbacksLock(m_vehicleExteriorColorCallbacksMutex);
    const auto vehicleExteriorColorCallbacks = m_vehicleExteriorColorCallbacks;
    vehicleExteriorColorCallbacksLock.unlock();
    for(const auto& callbackEntry: vehicleExteriorColorCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(VehicleExteriorColor);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToVehicleInteriorColorChanged(VehicleVehicleIdentificationVehicleInteriorColorPropertyCb callback)
{
    auto handleId = m_vehicleInteriorColorChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleInteriorColorCallbacksMutex);
    m_vehicleInteriorColorCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromVehicleInteriorColorChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleInteriorColorCallbacksMutex);
    m_vehicleInteriorColorCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishVehicleInteriorColorChanged(const std::string& VehicleInteriorColor) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onVehicleInteriorColorChanged(VehicleInteriorColor);
    }
    std::shared_lock<std::shared_timed_mutex> vehicleInteriorColorCallbacksLock(m_vehicleInteriorColorCallbacksMutex);
    const auto vehicleInteriorColorCallbacks = m_vehicleInteriorColorCallbacks;
    vehicleInteriorColorCallbacksLock.unlock();
    for(const auto& callbackEntry: vehicleInteriorColorCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(VehicleInteriorColor);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToVehicleInteriorTypeChanged(VehicleVehicleIdentificationVehicleInteriorTypePropertyCb callback)
{
    auto handleId = m_vehicleInteriorTypeChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleInteriorTypeCallbacksMutex);
    m_vehicleInteriorTypeCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromVehicleInteriorTypeChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_vehicleInteriorTypeCallbacksMutex);
    m_vehicleInteriorTypeCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishVehicleInteriorTypeChanged(const std::string& VehicleInteriorType) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onVehicleInteriorTypeChanged(VehicleInteriorType);
    }
    std::shared_lock<std::shared_timed_mutex> vehicleInteriorTypeCallbacksLock(m_vehicleInteriorTypeCallbacksMutex);
    const auto vehicleInteriorTypeCallbacks = m_vehicleInteriorTypeCallbacks;
    vehicleInteriorTypeCallbacksLock.unlock();
    for(const auto& callbackEntry: vehicleInteriorTypeCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(VehicleInteriorType);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToKnownVehicleDamagesChanged(VehicleVehicleIdentificationKnownVehicleDamagesPropertyCb callback)
{
    auto handleId = m_knownVehicleDamagesChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_knownVehicleDamagesCallbacksMutex);
    m_knownVehicleDamagesCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromKnownVehicleDamagesChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_knownVehicleDamagesCallbacksMutex);
    m_knownVehicleDamagesCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishKnownVehicleDamagesChanged(const std::string& KnownVehicleDamages) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onKnownVehicleDamagesChanged(KnownVehicleDamages);
    }
    std::shared_lock<std::shared_timed_mutex> knownVehicleDamagesCallbacksLock(m_knownVehicleDamagesCallbacksMutex);
    const auto knownVehicleDamagesCallbacks = m_knownVehicleDamagesCallbacks;
    knownVehicleDamagesCallbacksLock.unlock();
    for(const auto& callbackEntry: knownVehicleDamagesCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(KnownVehicleDamages);
        }
    }
}

long VehicleVehicleIdentificationPublisher::subscribeToOptionalExtrasChanged(VehicleVehicleIdentificationOptionalExtrasPropertyCb callback)
{
    auto handleId = m_optionalExtrasChangedCallbackNextId++;
    std::unique_lock<std::shared_timed_mutex> lock(m_optionalExtrasCallbacksMutex);
    m_optionalExtrasCallbacks[handleId] = callback;
    return handleId;
}

void VehicleVehicleIdentificationPublisher::unsubscribeFromOptionalExtrasChanged(long handleId)
{
    std::unique_lock<std::shared_timed_mutex> lock(m_optionalExtrasCallbacksMutex);
    m_optionalExtrasCallbacks.erase(handleId);
}

void VehicleVehicleIdentificationPublisher::publishOptionalExtrasChanged(const std::list<std::string>& OptionalExtras) const
{
    std::shared_lock<std::shared_timed_mutex> allChangesSubscribersLock(m_allChangesSubscribersMutex);
    const auto allChangesSubscribers = m_allChangesSubscribers;
    allChangesSubscribersLock.unlock();
    for(const auto& subscriber: allChangesSubscribers)
    {
        subscriber.get().onOptionalExtrasChanged(OptionalExtras);
    }
    std::shared_lock<std::shared_timed_mutex> optionalExtrasCallbacksLock(m_optionalExtrasCallbacksMutex);
    const auto optionalExtrasCallbacks = m_optionalExtrasCallbacks;
    optionalExtrasCallbacksLock.unlock();
    for(const auto& callbackEntry: optionalExtrasCallbacks)
    {
        if(callbackEntry.second)
        {
            callbackEntry.second(OptionalExtras);
        }
    }
}

