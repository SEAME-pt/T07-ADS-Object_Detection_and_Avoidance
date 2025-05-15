#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertrainelectricmotorenginecoolant.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainElectricMotorEngineCoolantPublisher.
 * Use this class to store clients of the Vehicle_Powertrain_ElectricMotor_EngineCoolant and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainElectricMotorEngineCoolantPublisher : public IVehiclePowertrainElectricMotorEngineCoolantPublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainElectricMotorEngineCoolantPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainElectricMotorEngineCoolantSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorEngineCoolantPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainElectricMotorEngineCoolantSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainElectricMotorEngineCoolantPublisher::subscribeToCapacityChanged
    */
    long subscribeToCapacityChanged(VehiclePowertrainElectricMotorEngineCoolantCapacityPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorEngineCoolantPublisher::subscribeToCapacityChanged
    */
    void unsubscribeFromCapacityChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainElectricMotorEngineCoolantPublisher::subscribeToTemperatureChanged
    */
    long subscribeToTemperatureChanged(VehiclePowertrainElectricMotorEngineCoolantTemperaturePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorEngineCoolantPublisher::subscribeToTemperatureChanged
    */
    void unsubscribeFromTemperatureChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainElectricMotorEngineCoolantPublisher::subscribeToLevelChanged
    */
    long subscribeToLevelChanged(VehiclePowertrainElectricMotorEngineCoolantLevelPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorEngineCoolantPublisher::subscribeToLevelChanged
    */
    void unsubscribeFromLevelChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainElectricMotorEngineCoolantPublisher::subscribeToLifeRemainingChanged
    */
    long subscribeToLifeRemainingChanged(VehiclePowertrainElectricMotorEngineCoolantLifeRemainingPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorEngineCoolantPublisher::subscribeToLifeRemainingChanged
    */
    void unsubscribeFromLifeRemainingChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainElectricMotorEngineCoolantPublisher::publishCapacityChanged
    */
    void publishCapacityChanged(float Capacity) const override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorEngineCoolantPublisher::publishTemperatureChanged
    */
    void publishTemperatureChanged(float Temperature) const override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorEngineCoolantPublisher::publishLevelChanged
    */
    void publishLevelChanged(Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum Level) const override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorEngineCoolantPublisher::publishLifeRemainingChanged
    */
    void publishLifeRemainingChanged(int32_t LifeRemaining) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_ElectricMotor_EngineCoolant
    std::vector<std::reference_wrapper<IVehiclePowertrainElectricMotorEngineCoolantSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the Capacity change.
    std::atomic<long> m_capacityChangedCallbackNextId {0};
    // Subscribed callbacks for the Capacity change.
    std::map<long, VehiclePowertrainElectricMotorEngineCoolantCapacityPropertyCb> m_capacityCallbacks;
    // Mutex for m_capacityCallbacks
    mutable std::shared_timed_mutex m_capacityCallbacksMutex;
    // Next free unique identifier to subscribe for the Temperature change.
    std::atomic<long> m_temperatureChangedCallbackNextId {0};
    // Subscribed callbacks for the Temperature change.
    std::map<long, VehiclePowertrainElectricMotorEngineCoolantTemperaturePropertyCb> m_temperatureCallbacks;
    // Mutex for m_temperatureCallbacks
    mutable std::shared_timed_mutex m_temperatureCallbacksMutex;
    // Next free unique identifier to subscribe for the Level change.
    std::atomic<long> m_levelChangedCallbackNextId {0};
    // Subscribed callbacks for the Level change.
    std::map<long, VehiclePowertrainElectricMotorEngineCoolantLevelPropertyCb> m_levelCallbacks;
    // Mutex for m_levelCallbacks
    mutable std::shared_timed_mutex m_levelCallbacksMutex;
    // Next free unique identifier to subscribe for the LifeRemaining change.
    std::atomic<long> m_lifeRemainingChangedCallbackNextId {0};
    // Subscribed callbacks for the LifeRemaining change.
    std::map<long, VehiclePowertrainElectricMotorEngineCoolantLifeRemainingPropertyCb> m_lifeRemainingCallbacks;
    // Mutex for m_lifeRemainingCallbacks
    mutable std::shared_timed_mutex m_lifeRemainingCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
