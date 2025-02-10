#pragma once

#include <future>
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/api/datastructs.api.h"

namespace Cpp {
namespace Vehicle {

class IVehiclePowertrainElectricMotorEngineCoolantSubscriber;
class IVehiclePowertrainElectricMotorEngineCoolantPublisher;

/**
*
* IVehiclePowertrainElectricMotorEngineCoolant provides an interface for
 *  - methods defined for your Vehicle_Powertrain_ElectricMotor_EngineCoolant 
 *  - property setters and getters for defined properties
 * The IVehiclePowertrainElectricMotorEngineCoolant also provides an interface to access a publisher IVehiclePowertrainElectricMotorEngineCoolantPublisher, a class used by IVehiclePowertrainElectricMotorEngineCoolantSubscriber clients.
 * The implementation should notify the publisher IVehiclePowertrainElectricMotorEngineCoolantPublisher about emitted signals or state changed. 
 * The publisher responsibility is to keep its clients informed about requested changes.
 * See also IVehiclePowertrainElectricMotorEngineCoolantSubscriber, IVehiclePowertrainElectricMotorEngineCoolantPublisher
 * and the example implementation Vehicle_Powertrain_ElectricMotor_EngineCoolant  or the
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainElectricMotorEngineCoolant
{
public:
    virtual ~IVehiclePowertrainElectricMotorEngineCoolant() = default;

    /**
    * Sets the value of the Capacity property.
    * @param Capacity Engine coolant capacity in liters.
    */
    virtual void setCapacity(float Capacity) = 0;
    /**
    * Gets the value of the Capacity property.
    * @return Engine coolant capacity in liters.
    */
    virtual float getCapacity() const = 0;

    /**
    * Sets the value of the Temperature property.
    * @param Temperature Engine coolant temperature.
    */
    virtual void setTemperature(float Temperature) = 0;
    /**
    * Gets the value of the Temperature property.
    * @return Engine coolant temperature.
    */
    virtual float getTemperature() const = 0;

    /**
    * Sets the value of the Level property.
    * @param Level Engine coolant level.
    */
    virtual void setLevel(Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum Level) = 0;
    /**
    * Gets the value of the Level property.
    * @return Engine coolant level.
    */
    virtual Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum getLevel() const = 0;

    /**
    * Sets the value of the LifeRemaining property.
    * @param LifeRemaining Remaining engine coolant life in seconds. Negative values can be used to indicate that lifetime has been exceeded.
    */
    virtual void setLifeRemaining(int32_t LifeRemaining) = 0;
    /**
    * Gets the value of the LifeRemaining property.
    * @return Remaining engine coolant life in seconds. Negative values can be used to indicate that lifetime has been exceeded.
    */
    virtual int32_t getLifeRemaining() const = 0;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_ElectricMotor_EngineCoolant changes and signal emission.
    * This function name doesn't follow the convention, because it is added to user defined interface,
    * to avoid potentially name clashes, it has the trailing underscore in the name.
    * @return The publisher for Vehicle_Powertrain_ElectricMotor_EngineCoolant.
    */
    virtual IVehiclePowertrainElectricMotorEngineCoolantPublisher& _getPublisher() const = 0;
};


/**
 * The IVehiclePowertrainElectricMotorEngineCoolantSubscriber contains functions to allow informing about signals or property changes of the IVehiclePowertrainElectricMotorEngineCoolant implementation.
 * The implementation for IVehiclePowertrainElectricMotorEngineCoolant should provide mechanism for subscription of the IVehiclePowertrainElectricMotorEngineCoolantSubscriber clients.
 * See IVehiclePowertrainElectricMotorEngineCoolantPublisher, which provides facilitation for this purpose.
 * The implementation for IVehiclePowertrainElectricMotorEngineCoolant should call the IVehiclePowertrainElectricMotorEngineCoolantSubscriber interface functions on either signal emit or property change.
 * You can use IVehiclePowertrainElectricMotorEngineCoolantSubscriber class to implement clients of the IVehiclePowertrainElectricMotorEngineCoolant or the network adapter - see Olink Server and Client example.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainElectricMotorEngineCoolantSubscriber
{
public:
    virtual ~IVehiclePowertrainElectricMotorEngineCoolantSubscriber() = default;
    /**
    * Called by the IVehiclePowertrainElectricMotorEngineCoolantPublisher when Capacity value has changed if subscribed for the Capacity change.
    * @param Capacity Engine coolant capacity in liters.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onCapacityChanged(float Capacity) = 0;
    /**
    * Called by the IVehiclePowertrainElectricMotorEngineCoolantPublisher when Temperature value has changed if subscribed for the Temperature change.
    * @param Temperature Engine coolant temperature.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onTemperatureChanged(float Temperature) = 0;
    /**
    * Called by the IVehiclePowertrainElectricMotorEngineCoolantPublisher when Level value has changed if subscribed for the Level change.
    * @param Level Engine coolant level.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onLevelChanged(Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum Level) = 0;
    /**
    * Called by the IVehiclePowertrainElectricMotorEngineCoolantPublisher when LifeRemaining value has changed if subscribed for the LifeRemaining change.
    * @param LifeRemaining Remaining engine coolant life in seconds. Negative values can be used to indicate that lifetime has been exceeded.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual void onLifeRemainingChanged(int32_t LifeRemaining) = 0;
};

/** Callback for changes of Capacity */
using VehiclePowertrainElectricMotorEngineCoolantCapacityPropertyCb = std::function<void(float Capacity)>;
/** Callback for changes of Temperature */
using VehiclePowertrainElectricMotorEngineCoolantTemperaturePropertyCb = std::function<void(float Temperature)>;
/** Callback for changes of Level */
using VehiclePowertrainElectricMotorEngineCoolantLevelPropertyCb = std::function<void(Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum Level)>;
/** Callback for changes of LifeRemaining */
using VehiclePowertrainElectricMotorEngineCoolantLifeRemainingPropertyCb = std::function<void(int32_t LifeRemaining)>;


/**
 * The IVehiclePowertrainElectricMotorEngineCoolantPublisher provides an api for clients to subscribe to or unsubscribe from a signal emission 
 * or a property change.
 * Implement this interface to keep track of clients of your IVehiclePowertrainElectricMotorEngineCoolant implementation.
 * The publisher provides two independent methods of subscription
 *  - subscribing with a IVehiclePowertrainElectricMotorEngineCoolantSubscriber objects - for all of the changes
 *  - subscribing any object for single type of change property or a signal
 * The publish functions needs to be called by implementation of the IIVehiclePowertrainElectricMotorEngineCoolant on each state changed or signal emitted
 * to notify all the subscribers about this change.
 */
class CPP_VEHICLE_EXPORT IVehiclePowertrainElectricMotorEngineCoolantPublisher
{
public:
    virtual ~IVehiclePowertrainElectricMotorEngineCoolantPublisher() = default;

    /**
    * Use this function to subscribe for any change of the Vehicle_Powertrain_ElectricMotor_EngineCoolant.
    * Subscriber will be informed of any emitted signal and any property changes.
    * This is parallel notification system to single subscription. If you will subscribe also for a single change
    * your subscriber will be informed twice about that change, one for each subscription mechanism.
    * @param IVehiclePowertrainElectricMotorEngineCoolantSubscriber which is subscribed in this function to any change of the Vehicle_Powertrain_ElectricMotor_EngineCoolant.
    */
    virtual void subscribeToAllChanges(IVehiclePowertrainElectricMotorEngineCoolantSubscriber& subscriber) = 0;
    /**
    * Use this function to remove subscription to all of the changes of the Vehicle_Powertrain_ElectricMotor_EngineCoolant.
    * Not all subscriptions will be removed, the ones made separately for single signal or property change stay intact.
    * Make sure to remove them.
    * @param IVehiclePowertrainElectricMotorEngineCoolantSubscriber which subscription for any change of the Vehicle_Powertrain_ElectricMotor_EngineCoolant is removed.
    */
    virtual void unsubscribeFromAllChanges(IVehiclePowertrainElectricMotorEngineCoolantSubscriber& subscriber) = 0;

    /**
    * Use this function to subscribe for Capacity value changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorEngineCoolantSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainElectricMotorEngineCoolantCapacityPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToCapacityChanged(VehiclePowertrainElectricMotorEngineCoolantCapacityPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Capacity property changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorEngineCoolantSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromCapacityChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Temperature value changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorEngineCoolantSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainElectricMotorEngineCoolantTemperaturePropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToTemperatureChanged(VehiclePowertrainElectricMotorEngineCoolantTemperaturePropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Temperature property changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorEngineCoolantSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromTemperatureChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for Level value changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorEngineCoolantSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainElectricMotorEngineCoolantLevelPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToLevelChanged(VehiclePowertrainElectricMotorEngineCoolantLevelPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from Level property changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorEngineCoolantSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromLevelChanged(long handleId) = 0;

    /**
    * Use this function to subscribe for LifeRemaining value changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorEngineCoolantSubscriber interface, you will get two notifications, one for each subscription mechanism.
    * @param VehiclePowertrainElectricMotorEngineCoolantLifeRemainingPropertyCb callback that will be executed on each change of the property.
    * Make sure to remove subscription before the callback becomes invalid.
    * @return subscription token for the subscription removal.
    *
    * @warning the subscribed function shall not be blocking and must return immediately!
    */
    virtual long subscribeToLifeRemainingChanged(VehiclePowertrainElectricMotorEngineCoolantLifeRemainingPropertyCb callback) = 0;
    /**
    * Use this function to unsubscribe from LifeRemaining property changes.
    * If your subscriber uses subscription with IVehiclePowertrainElectricMotorEngineCoolantSubscriber interface, you will be still informed about this change,
    * as those are two independent subscription mechanisms.
    * @param subscription token received on subscription.
    */
    virtual void unsubscribeFromLifeRemainingChanged(long handleId) = 0;

    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_ElectricMotor_EngineCoolant implementation when property Capacity changes.
    * @param The new value of Capacity.
    * Engine coolant capacity in liters.
    */
    virtual void publishCapacityChanged(float Capacity) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_ElectricMotor_EngineCoolant implementation when property Temperature changes.
    * @param The new value of Temperature.
    * Engine coolant temperature.
    */
    virtual void publishTemperatureChanged(float Temperature) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_ElectricMotor_EngineCoolant implementation when property Level changes.
    * @param The new value of Level.
    * Engine coolant level.
    */
    virtual void publishLevelChanged(Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum Level) const = 0;
    /**
    * Publishes the property changed to all subscribed clients.
    * Needs to be invoked by the Vehicle_Powertrain_ElectricMotor_EngineCoolant implementation when property LifeRemaining changes.
    * @param The new value of LifeRemaining.
    * Remaining engine coolant life in seconds. Negative values can be used to indicate that lifetime has been exceeded.
    */
    virtual void publishLifeRemainingChanged(int32_t LifeRemaining) const = 0;
};


} // namespace Vehicle
} // namespace Cpp
