#pragma once

#include "vehicle/generated/api/datastructs.api.h"
#include "vehicle/generated/api/vehiclepowertrainelectricmotor.api.h"
#include "vehicle/generated/api/common.h"

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/**
 * The implementation of a VehiclePowertrainElectricMotorPublisher.
 * Use this class to store clients of the Vehicle_Powertrain_ElectricMotor and inform them about the change
 * on call of the appropriate publish function.
 *
 * @warning This class is thread safe, but the subscribed classes or functions are not protected.
 */
class CPP_VEHICLE_EXPORT VehiclePowertrainElectricMotorPublisher : public IVehiclePowertrainElectricMotorPublisher
{
public:
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToAllChanges
    */
    void subscribeToAllChanges(IVehiclePowertrainElectricMotorSubscriber& subscriber) override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::unsubscribeFromAllChanges
    */
    void unsubscribeFromAllChanges(IVehiclePowertrainElectricMotorSubscriber& subscriber) override;

    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToEngineCodeChanged
    */
    long subscribeToEngineCodeChanged(VehiclePowertrainElectricMotorEngineCodePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToEngineCodeChanged
    */
    void unsubscribeFromEngineCodeChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToMaxPowerChanged
    */
    long subscribeToMaxPowerChanged(VehiclePowertrainElectricMotorMaxPowerPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToMaxPowerChanged
    */
    void unsubscribeFromMaxPowerChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToMaxTorqueChanged
    */
    long subscribeToMaxTorqueChanged(VehiclePowertrainElectricMotorMaxTorquePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToMaxTorqueChanged
    */
    void unsubscribeFromMaxTorqueChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToMaxRegenPowerChanged
    */
    long subscribeToMaxRegenPowerChanged(VehiclePowertrainElectricMotorMaxRegenPowerPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToMaxRegenPowerChanged
    */
    void unsubscribeFromMaxRegenPowerChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToMaxRegenTorqueChanged
    */
    long subscribeToMaxRegenTorqueChanged(VehiclePowertrainElectricMotorMaxRegenTorquePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToMaxRegenTorqueChanged
    */
    void unsubscribeFromMaxRegenTorqueChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToSpeedChanged
    */
    long subscribeToSpeedChanged(VehiclePowertrainElectricMotorSpeedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToSpeedChanged
    */
    void unsubscribeFromSpeedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToTimeInUseChanged
    */
    long subscribeToTimeInUseChanged(VehiclePowertrainElectricMotorTimeInUsePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToTimeInUseChanged
    */
    void unsubscribeFromTimeInUseChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToTemperatureChanged
    */
    long subscribeToTemperatureChanged(VehiclePowertrainElectricMotorTemperaturePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToTemperatureChanged
    */
    void unsubscribeFromTemperatureChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToCoolantTemperatureDeprecatedChanged
    */
    long subscribeToCoolantTemperatureDeprecatedChanged(VehiclePowertrainElectricMotorCoolantTemperatureDeprecatedPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToCoolantTemperatureDeprecatedChanged
    */
    void unsubscribeFromCoolantTemperatureDeprecatedChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToPowerChanged
    */
    long subscribeToPowerChanged(VehiclePowertrainElectricMotorPowerPropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToPowerChanged
    */
    void unsubscribeFromPowerChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToTorqueChanged
    */
    long subscribeToTorqueChanged(VehiclePowertrainElectricMotorTorquePropertyCb callback) override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::subscribeToTorqueChanged
    */
    void unsubscribeFromTorqueChanged(long handleId) override;

    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::publishEngineCodeChanged
    */
    void publishEngineCodeChanged(const std::string& EngineCode) const override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::publishMaxPowerChanged
    */
    void publishMaxPowerChanged(int32_t MaxPower) const override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::publishMaxTorqueChanged
    */
    void publishMaxTorqueChanged(int32_t MaxTorque) const override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::publishMaxRegenPowerChanged
    */
    void publishMaxRegenPowerChanged(int32_t MaxRegenPower) const override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::publishMaxRegenTorqueChanged
    */
    void publishMaxRegenTorqueChanged(int32_t MaxRegenTorque) const override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::publishSpeedChanged
    */
    void publishSpeedChanged(int32_t Speed) const override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::publishTimeInUseChanged
    */
    void publishTimeInUseChanged(float TimeInUse) const override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::publishTemperatureChanged
    */
    void publishTemperatureChanged(float Temperature) const override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::publishCoolantTemperatureDeprecatedChanged
    */
    void publishCoolantTemperatureDeprecatedChanged(float CoolantTemperature_Deprecated) const override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::publishPowerChanged
    */
    void publishPowerChanged(int32_t Power) const override;
    /**
    * Implementation of IVehiclePowertrainElectricMotorPublisher::publishTorqueChanged
    */
    void publishTorqueChanged(int32_t Torque) const override;
private:
    // Subscribers informed about any property change or signal emitted in Vehicle_Powertrain_ElectricMotor
    std::vector<std::reference_wrapper<IVehiclePowertrainElectricMotorSubscriber>> m_allChangesSubscribers;
    // Mutex for m_allChangesSubscribers
    mutable std::shared_timed_mutex m_allChangesSubscribersMutex;
    // Next free unique identifier to subscribe for the EngineCode change.
    std::atomic<long> m_engineCodeChangedCallbackNextId {0};
    // Subscribed callbacks for the EngineCode change.
    std::map<long, VehiclePowertrainElectricMotorEngineCodePropertyCb> m_engineCodeCallbacks;
    // Mutex for m_engineCodeCallbacks
    mutable std::shared_timed_mutex m_engineCodeCallbacksMutex;
    // Next free unique identifier to subscribe for the MaxPower change.
    std::atomic<long> m_maxPowerChangedCallbackNextId {0};
    // Subscribed callbacks for the MaxPower change.
    std::map<long, VehiclePowertrainElectricMotorMaxPowerPropertyCb> m_maxPowerCallbacks;
    // Mutex for m_maxPowerCallbacks
    mutable std::shared_timed_mutex m_maxPowerCallbacksMutex;
    // Next free unique identifier to subscribe for the MaxTorque change.
    std::atomic<long> m_maxTorqueChangedCallbackNextId {0};
    // Subscribed callbacks for the MaxTorque change.
    std::map<long, VehiclePowertrainElectricMotorMaxTorquePropertyCb> m_maxTorqueCallbacks;
    // Mutex for m_maxTorqueCallbacks
    mutable std::shared_timed_mutex m_maxTorqueCallbacksMutex;
    // Next free unique identifier to subscribe for the MaxRegenPower change.
    std::atomic<long> m_maxRegenPowerChangedCallbackNextId {0};
    // Subscribed callbacks for the MaxRegenPower change.
    std::map<long, VehiclePowertrainElectricMotorMaxRegenPowerPropertyCb> m_maxRegenPowerCallbacks;
    // Mutex for m_maxRegenPowerCallbacks
    mutable std::shared_timed_mutex m_maxRegenPowerCallbacksMutex;
    // Next free unique identifier to subscribe for the MaxRegenTorque change.
    std::atomic<long> m_maxRegenTorqueChangedCallbackNextId {0};
    // Subscribed callbacks for the MaxRegenTorque change.
    std::map<long, VehiclePowertrainElectricMotorMaxRegenTorquePropertyCb> m_maxRegenTorqueCallbacks;
    // Mutex for m_maxRegenTorqueCallbacks
    mutable std::shared_timed_mutex m_maxRegenTorqueCallbacksMutex;
    // Next free unique identifier to subscribe for the Speed change.
    std::atomic<long> m_speedChangedCallbackNextId {0};
    // Subscribed callbacks for the Speed change.
    std::map<long, VehiclePowertrainElectricMotorSpeedPropertyCb> m_speedCallbacks;
    // Mutex for m_speedCallbacks
    mutable std::shared_timed_mutex m_speedCallbacksMutex;
    // Next free unique identifier to subscribe for the TimeInUse change.
    std::atomic<long> m_timeInUseChangedCallbackNextId {0};
    // Subscribed callbacks for the TimeInUse change.
    std::map<long, VehiclePowertrainElectricMotorTimeInUsePropertyCb> m_timeInUseCallbacks;
    // Mutex for m_timeInUseCallbacks
    mutable std::shared_timed_mutex m_timeInUseCallbacksMutex;
    // Next free unique identifier to subscribe for the Temperature change.
    std::atomic<long> m_temperatureChangedCallbackNextId {0};
    // Subscribed callbacks for the Temperature change.
    std::map<long, VehiclePowertrainElectricMotorTemperaturePropertyCb> m_temperatureCallbacks;
    // Mutex for m_temperatureCallbacks
    mutable std::shared_timed_mutex m_temperatureCallbacksMutex;
    // Next free unique identifier to subscribe for the CoolantTemperatureDeprecated change.
    std::atomic<long> m_coolantTemperatureDeprecatedChangedCallbackNextId {0};
    // Subscribed callbacks for the CoolantTemperatureDeprecated change.
    std::map<long, VehiclePowertrainElectricMotorCoolantTemperatureDeprecatedPropertyCb> m_coolantTemperatureDeprecatedCallbacks;
    // Mutex for m_coolantTemperatureDeprecatedCallbacks
    mutable std::shared_timed_mutex m_coolantTemperatureDeprecatedCallbacksMutex;
    // Next free unique identifier to subscribe for the Power change.
    std::atomic<long> m_powerChangedCallbackNextId {0};
    // Subscribed callbacks for the Power change.
    std::map<long, VehiclePowertrainElectricMotorPowerPropertyCb> m_powerCallbacks;
    // Mutex for m_powerCallbacks
    mutable std::shared_timed_mutex m_powerCallbacksMutex;
    // Next free unique identifier to subscribe for the Torque change.
    std::atomic<long> m_torqueChangedCallbackNextId {0};
    // Subscribed callbacks for the Torque change.
    std::map<long, VehiclePowertrainElectricMotorTorquePropertyCb> m_torqueCallbacks;
    // Mutex for m_torqueCallbacks
    mutable std::shared_timed_mutex m_torqueCallbacksMutex;
};

} // namespace Vehicle
} // namespace Cpp
