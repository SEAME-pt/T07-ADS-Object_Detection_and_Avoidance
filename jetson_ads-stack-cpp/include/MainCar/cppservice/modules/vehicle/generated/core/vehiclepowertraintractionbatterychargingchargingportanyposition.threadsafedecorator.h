
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition interface.
* @see Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition> testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition = std::make_unique<VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionThreadSafeDecorator>(std::make_shared<VehiclePowertrainTractionBatteryChargingChargingPortAnyPosition>());

// Thread safe access
auto supportedInletTypes = testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition->getSupportedInletTypes();
testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition->setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC);
auto isFlapOpen = testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition->getIsFlapOpen();
testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition->setIsFlapOpen(false);
auto isChargingCableConnected = testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition->getIsChargingCableConnected();
testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition->setIsChargingCableConnected(false);
auto isChargingCableLocked = testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition->getIsChargingCableLocked();
testVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition->setIsChargingCableLocked(false);
\endcode
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionThreadSafeDecorator : public IVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition object to make thread safe.
    */
    explicit VehiclePowertrainTractionBatteryChargingChargingPortAnyPositionThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition> impl);

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition implementation. */
    void setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum SupportedInletTypes) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition implementation. */
    Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum getSupportedInletTypes() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition implementation. */
    void setIsFlapOpen(bool IsFlapOpen) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition implementation. */
    bool getIsFlapOpen() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition implementation. */
    void setIsChargingCableConnected(bool IsChargingCableConnected) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition implementation. */
    bool getIsChargingCableConnected() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition implementation. */
    void setIsChargingCableLocked(bool IsChargingCableLocked) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition implementation. */
    bool getIsChargingCableLocked() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition.
    */
    IVehiclePowertrainTractionBatteryChargingChargingPortAnyPositionPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition object which is guarded */
    std::shared_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortAnyPosition> m_impl;
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
