
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehiclePowertrainTractionBatteryChargingChargingPortRearLeftThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft interface.
* @see Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortRearLeft> testVehiclePowertrainTractionBatteryChargingChargingPortRearLeft = std::make_unique<VehiclePowertrainTractionBatteryChargingChargingPortRearLeftThreadSafeDecorator>(std::make_shared<VehiclePowertrainTractionBatteryChargingChargingPortRearLeft>());

// Thread safe access
auto supportedInletTypes = testVehiclePowertrainTractionBatteryChargingChargingPortRearLeft->getSupportedInletTypes();
testVehiclePowertrainTractionBatteryChargingChargingPortRearLeft->setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC);
auto isFlapOpen = testVehiclePowertrainTractionBatteryChargingChargingPortRearLeft->getIsFlapOpen();
testVehiclePowertrainTractionBatteryChargingChargingPortRearLeft->setIsFlapOpen(false);
auto isChargingCableConnected = testVehiclePowertrainTractionBatteryChargingChargingPortRearLeft->getIsChargingCableConnected();
testVehiclePowertrainTractionBatteryChargingChargingPortRearLeft->setIsChargingCableConnected(false);
auto isChargingCableLocked = testVehiclePowertrainTractionBatteryChargingChargingPortRearLeft->getIsChargingCableLocked();
testVehiclePowertrainTractionBatteryChargingChargingPortRearLeft->setIsChargingCableLocked(false);
\endcode
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingChargingPortRearLeftThreadSafeDecorator : public IVehiclePowertrainTractionBatteryChargingChargingPortRearLeft
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft object to make thread safe.
    */
    explicit VehiclePowertrainTractionBatteryChargingChargingPortRearLeftThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortRearLeft> impl);

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft implementation. */
    void setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum SupportedInletTypes) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft implementation. */
    Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum getSupportedInletTypes() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft implementation. */
    void setIsFlapOpen(bool IsFlapOpen) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft implementation. */
    bool getIsFlapOpen() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft implementation. */
    void setIsChargingCableConnected(bool IsChargingCableConnected) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft implementation. */
    bool getIsChargingCableConnected() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft implementation. */
    void setIsChargingCableLocked(bool IsChargingCableLocked) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft implementation. */
    bool getIsChargingCableLocked() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft.
    */
    IVehiclePowertrainTractionBatteryChargingChargingPortRearLeftPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft object which is guarded */
    std::shared_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortRearLeft> m_impl;
    // Mutex for supportedInletTypes property
    mutable std::shared_timed_mutex m_supportedInletTypesMutex;
    // Mutex for isFlapOpen property
    mutable std::shared_timed_mutex m_isFlapOpenMutex;
    // Mutex for isChargingCableConnected property
    mutable std::shared_timed_mutex m_isChargingCableConnectedMutex;
    // Mutex for isChargingCableLocked property
    mutable std::shared_timed_mutex m_isChargingCableLockedMutex;
};
} // namespace Vehicle
} // namespace Cpp
