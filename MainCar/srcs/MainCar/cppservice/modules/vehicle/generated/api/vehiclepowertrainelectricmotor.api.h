#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainElectricMotorSubscriber;
class IVehiclePowertrainElectricMotorPublisher;

/**
*
* IVehiclePowertrainElectricMotor provides an interface for
 *  - methods defined for your Vehicle_Powertrain_ElectricMotor 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrainElectricMotor also provides an interface to access a publisher IVehiclePowertrainElectricMotorPublisher, a class used by IVehiclePowertrainElectricMotorSubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainElectricMotorPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainElectricMotorSubscriber, IVehiclePowertrainElectricMotorPublisher
 * and the example implementation Vehicle_Powertrain_ElectricMotor  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainElectricMotor
{
public:
    virtual ~IVehiclePowertrainElectricMotor() = default;

    /**
    * Sets the value of the EngineCode property.
    * @param EngineCode Engine code designation, as specified by vehicle manufacturer.
    */
    virtual void setEngineCode(const std::string& EngineCode) = 0;
    /**
    * Gets the value of the EngineCode property.
    * @return Engine code designation, as specified by vehicle manufacturer.
    */
    virtual const std::string& getEngineCode() const = 0;

    /**
    * Sets the value of the MaxPower property.
    * @param MaxPower Peak power, in kilowatts, that motor(s) can generate.
    */
    virtual void setMaxPower(int32_t MaxPower) = 0;
    /**
    * Gets the value of the MaxPower property.
    * @return Peak power, in kilowatts, that motor(s) can generate.
    */
    virtual int32_t getMaxPower() const = 0;

    /**
    * Sets the value of the MaxTorque property.
    * @param MaxTorque Peak power, in newton meter, that the motor(s) can generate.
    */
    virtual void setMaxTorque(int32_t MaxTorque) = 0;
    /**
    * Gets the value of the MaxTorque property.
    * @return Peak power, in newton meter, that the motor(s) can generate.
    */
    virtual int32_t getMaxTorque() const = 0;

    /**
    * Sets the value of the MaxRegenPower property.
    * @param MaxRegenPower Peak regen/brake power, in kilowatts, that motor(s) can generate.
    */
    virtual void setMaxRegenPower(int32_t MaxRegenPower) = 0;
    /**
    * Gets the value of the MaxRegenPower property.
    * @return Peak regen/brake power, in kilowatts, that motor(s) can generate.
    */
    virtual int32_t getMaxRegenPower() const = 0;

    /**
    * Sets the value of the MaxRegenTorque property.
    * @param MaxRegenTorque Peak regen/brake torque, in newton meter, that the motor(s) can generate.
    */
    virtual void setMaxRegenTorque(int32_t MaxRegenTorque) = 0;
    /**
    * Gets the value of the MaxRegenTorque property.
    * @return Peak regen/brake torque, in newton meter, that the motor(s) can generate.
    */
    virtual int32_t getMaxRegenTorque() const = 0;

    /**
    * Sets the value of the Speed property.
    * @param Speed Motor rotational speed measured as rotations per minute. Negative values indicate reverse driving mode.
    */
    virtual void setSpeed(int32_t Speed) = 0;
    /**
    * Gets the value of the Speed property.
    * @return Motor rotational speed measured as rotations per minute. Negative values indicate reverse driving mode.
    */
    virtual int32_t getSpeed() const = 0;

    /**
    * Sets the value of the TimeInUse property.
    * @param TimeInUse Accumulated time during engine lifetime when the vehicule state's is 'READY'.
    */
    virtual void setTimeInUse(float TimeInUse) = 0;
    /**
    * Gets the value of the TimeInUse property.
    * @return Accumulated time during engine lifetime when the vehicule state's is 'READY'.
    */
    virtual float getTimeInUse() const = 0;

    /**
    * Sets the value of the Temperature property.
    * @param Temperature Motor temperature.
    */
    virtual void setTemperature(float Temperature) = 0;
    /**
    * Gets the value of the Temperature property.
    * @return Motor temperature.
    */
    virtual float getTemperature() const = 0;

    /**
    * Sets the value of the CoolantTemperature_Deprecated property.
    * @param CoolantTemperature_Deprecated Motor coolant temperature (if applicable).
    */
    virtual void setCoolantTemperatureDeprecated(float CoolantTemperature_Deprecated) = 0;
    /**
    * Gets the value of the CoolantTemperature_Deprecated property.
    * @return Motor coolant temperature (if applicable).
    */
    virtual float getCoolantTemperatureDeprecated() const = 0;

    /**
    * Sets the value of the Power property.
    * @param Power Current motor power output. Negative values indicate regen mode.
    */
    virtual void setPower(int32_t Power) = 0;
    /**
    * Gets the value of the Power property.
    * @return Current motor power output. Negative values indicate regen mode.
    */
    virtual int32_t getPower() const = 0;

    /**
    * Sets the value of the Torque property.
    * @param Torque Current motor torque. Negative values indicate regen mode.
    */
    virtual void setTorque(int32_t Torque) = 0;
    /**
    * Gets the value of the Torque property.
    * @return Current motor torque. Negative values indicate regen mode.
    */
    virtual int32_t getTorque() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_ElectricMotor changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain_ElectricMotor.
    */
    virtual IVehiclePowertrainElectricMotorPublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainElectricMotorSubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrainElectricMotor implementation.
 * The implementation for IVehiclePowertrainElectricMotor should provide mechanism for subscription of the IVehiclePowertrainElectricMotorSubscriber clients.
 * See IVehiclePowertrainElectricMotorPublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrainElectricMotor should call the IVehiclePowertrainElectricMotorSubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainElectricMotorSubscriber class to implement clients of the IVehiclePowertrainElectricMotor or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainElectricMotorSubscriber
{
public:
    virtual ~IVehiclePowertrainElectricMotorSubscriber() = default;
    /**
    * Called by the IVehiclePowertrainElectricMotorPublisher when EngineCode value has changed if subscribed for the EngineCode change.
    * @param EngineCode Engine code designation, as specified by vehicle manufacturer.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onEngineCodeChanged(const std::string& EngineCode) = 0;
    /**
    * Called by the IVehiclePowertrainElectricMotorPublisher when MaxPower value has changed if subscribed for the MaxPower change.
    * @param MaxPower Peak power, in kilowatts, that motor(s) can generate.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onMaxPowerChanged(int32_t MaxPower) = 0;
    /**
    * Called by the IVehiclePowertrainElectricMotorPublisher when MaxTorque value has changed if subscribed for the MaxTorque change.
    * @param MaxTorque Peak power, in newton meter, that the motor(s) can generate.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onMaxTorqueChanged(int32_t MaxTorque) = 0;
    /**
    * Called by the IVehiclePowertrainElectricMotorPublisher when MaxRegenPower value has changed if subscribed for the MaxRegenPower change.
    * @param MaxRegenPower Peak regen/brake power, in kilowatts, that motor(s) can generate.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onMaxRegenPowerChanged(int32_t MaxRegenPower) = 0;
    /**
    * Called by the IVehiclePowertrainElectricMotorPublisher when MaxRegenTorque value has changed if subscribed for the MaxRegenTorque change.
    * @param MaxRegenTorque Peak regen/brake torque, in newton meter, that the motor(s) can generate.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onMaxRegenTorqueChanged(int32_t MaxRegenTorque) = 0;
    /**
    * Called by the IVehiclePowertrainElectricMotorPublisher when Speed value has changed if subscribed for the Speed change.
    * @param Speed Motor rotational speed measured as rotations per minute. Negative values indicate reverse driving mode.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onSpeedChanged(int32_t Speed) = 0;
    /**
    * Called by the IVehiclePowertrainElectricMotorPublisher when TimeInUse value has changed if subscribed for the TimeInUse change.
    * @param TimeInUse Accumulated time during engine lifetime when the vehicule state's is 'READY'.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTimeInUseChanged(float TimeInUse) = 0;
    /**
    * Called by the IVehiclePowertrainElectricMotorPublisher when Temperature value has changed if subscribed for the Temperature change.
    * @param Temperature Motor temperature.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTemperatureChanged(float Temperature) = 0;
    /**
    * Called by the IVehiclePowertrainElectricMotorPublisher when CoolantTemperature_Deprecated value has changed if subscribed for the CoolantTemperature_Deprecated change.
    * @param CoolantTemperature_Deprecated Motor coolant temperature (if applicable).
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onCoolantTemperatureDeprecatedChanged(float CoolantTemperature_Deprecated) = 0;
    /**
    * Called by the IVehiclePowertrainElectricMotorPublisher when Power value has changed if subscribed for the Power change.
    * @param Power Current motor power output. Negative values indicate regen mode.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onPowerChanged(int32_t Power) = 0;
    /**
    * Called by the IVehiclePowertrainElectricMotorPublisher when Torque value has changed if subscribed for the Torque change.
    * @param Torque Current motor torque. Negative values indicate regen mode.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTorqueChanged(int32_t Torque) = 0;
};

/** Callback for changes of EngineCode */
using VehiclePowertrainElectricMotorEngineCodePropertyCb = std::function<void(const std::string& EngineCode)>;
/** Callback for changes of MaxPower */
using VehiclePowertrainElectricMotorMaxPowerPropertyCb = std::function<void(int32_t MaxPower)>;
/** Callback for changes of MaxTorque */
using VehiclePowertrainElectricMotorMaxTorquePropertyCb = std::function<void(int32_t MaxTorque)>;
/** Callback for changes of MaxRegenPower */
using VehiclePowertrainElectricMotorMaxRegenPowerPropertyCb = std::function<void(int32_t MaxRegenPower)>;
/** Callback for changes of MaxRegenTorque */
using VehiclePowertrainElectricMotorMaxRegenTorquePropertyCb = std::function<void(int32_t MaxRegenTorque)>;
/** Callback for changes of Speed */
using VehiclePowertrainElectricMotorSpeedPropertyCb = std::function<void(int32_t Speed)>;
/** Callback for changes of TimeInUse */
using VehiclePowertrainElectricMotorTimeInUsePropertyCb = std::function<void(float TimeInUse)>;
/** Callback for changes of Temperature */
using VehiclePowertrainElectricMotorTemperaturePropertyCb = std::function<void(float Temperature)>;
/** Callback for changes of CoolantTemperature_Deprecated */
using VehiclePowertrainElectricMotorCoolantTemperatureDeprecatedPropertyCb = std::function<void(float CoolantTemperature_Deprecated)>;
/** Callback for changes of Power */
using VehiclePowertrainElectricMotorPowerPropertyCb = std::function<void(int32_t Power)>;
/** Callback for changes of Torque */
using VehiclePowertrainElectricMotorTorquePropertyCb = std::function<void(int32_t Torque)>;


/**
 * The IVehiclePowertrainElectricMotorPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrainElectricMotor implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainElectricMotorSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrainElectricMotor on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainElectricMotorPublisher
{
public:
    virtual ~IVehiclePowertrainElectricMotorPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain_ElectricMotor.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainElectricMotorSubscriber which is subscribed in this function to any change of the Vehicle_Powertrain_ElectricMotor.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainElectricMotorSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain_ElectricMotor.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainElectricMotorSubscriber which subscription for any change of the Vehicle_Powertrain_ElectricMotor is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainElectricMotorSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for EngineCode value changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainElectricMotorEngineCodePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToEngineCodeChanged(VehiclePowertrainElectricMotorEngineCodePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from EngineCode property changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromEngineCodeChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for MaxPower value changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainElectricMotorMaxPowerPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToMaxPowerChanged(VehiclePowertrainElectricMotorMaxPowerPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from MaxPower property changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromMaxPowerChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for MaxTorque value changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainElectricMotorMaxTorquePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToMaxTorqueChanged(VehiclePowertrainElectricMotorMaxTorquePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from MaxTorque property changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromMaxTorqueChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for MaxRegenPower value changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainElectricMotorMaxRegenPowerPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToMaxRegenPowerChanged(VehiclePowertrainElectricMotorMaxRegenPowerPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from MaxRegenPower property changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromMaxRegenPowerChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for MaxRegenTorque value changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainElectricMotorMaxRegenTorquePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToMaxRegenTorqueChanged(VehiclePowertrainElectricMotorMaxRegenTorquePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from MaxRegenTorque property changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromMaxRegenTorqueChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Speed value changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainElectricMotorSpeedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToSpeedChanged(VehiclePowertrainElectricMotorSpeedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Speed property changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromSpeedChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for TimeInUse value changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainElectricMotorTimeInUsePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTimeInUseChanged(VehiclePowertrainElectricMotorTimeInUsePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from TimeInUse property changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTimeInUseChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Temperature value changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainElectricMotorTemperaturePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTemperatureChanged(VehiclePowertrainElectricMotorTemperaturePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Temperature property changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTemperatureChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for CoolantTemperature_Deprecated value changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainElectricMotorCoolantTemperatureDeprecatedPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToCoolantTemperatureDeprecatedChanged(VehiclePowertrainElectricMotorCoolantTemperatureDeprecatedPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from CoolantTemperature_Deprecated property changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromCoolantTemperatureDeprecatedChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Power value changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainElectricMotorPowerPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToPowerChanged(VehiclePowertrainElectricMotorPowerPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Power property changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromPowerChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Torque value changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainElectricMotorTorquePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTorqueChanged(VehiclePowertrainElectricMotorTorquePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Torque property changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTorqueChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_ElectricMotor implementation when property EngineCode changes.
    * @param The new value of EngineCode.
    * Engine code designation, as specified by vehicle manufacturer.
    */
    virtual void publishEngineCodeChanged(const std::string& EngineCode) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_ElectricMotor implementation when property MaxPower changes.
    * @param The new value of MaxPower.
    * Peak power, in kilowatts, that motor(s) can generate.
    */
    virtual void publishMaxPowerChanged(int32_t MaxPower) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_ElectricMotor implementation when property MaxTorque changes.
    * @param The new value of MaxTorque.
    * Peak power, in newton meter, that the motor(s) can generate.
    */
    virtual void publishMaxTorqueChanged(int32_t MaxTorque) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_ElectricMotor implementation when property MaxRegenPower changes.
    * @param The new value of MaxRegenPower.
    * Peak regen/brake power, in kilowatts, that motor(s) can generate.
    */
    virtual void publishMaxRegenPowerChanged(int32_t MaxRegenPower) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_ElectricMotor implementation when property MaxRegenTorque changes.
    * @param The new value of MaxRegenTorque.
    * Peak regen/brake torque, in newton meter, that the motor(s) can generate.
    */
    virtual void publishMaxRegenTorqueChanged(int32_t MaxRegenTorque) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_ElectricMotor implementation when property Speed changes.
    * @param The new value of Speed.
    * Motor rotational speed measured as rotations per minute. Negative values indicate reverse driving mode.
    */
    virtual void publishSpeedChanged(int32_t Speed) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_ElectricMotor implementation when property TimeInUse changes.
    * @param The new value of TimeInUse.
    * Accumulated time during engine lifetime when the vehicule state's is 'READY'.
    */
    virtual void publishTimeInUseChanged(float TimeInUse) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_ElectricMotor implementation when property Temperature changes.
    * @param The new value of Temperature.
    * Motor temperature.
    */
    virtual void publishTemperatureChanged(float Temperature) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_ElectricMotor implementation when property CoolantTemperature_Deprecated changes.
    * @param The new value of CoolantTemperature_Deprecated.
    * Motor coolant temperature (if applicable).
    */
    virtual void publishCoolantTemperatureDeprecatedChanged(float CoolantTemperature_Deprecated) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_ElectricMotor implementation when property Power changes.
    * @param The new value of Power.
    * Current motor power output. Negative values indicate regen mode.
    */
    virtual void publishPowerChanged(int32_t Power) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_ElectricMotor implementation when property Torque changes.
    * @param The new value of Torque.
    * Current motor torque. Negative values indicate regen mode.
    */
    virtual void publishTorqueChanged(int32_t Torque) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
