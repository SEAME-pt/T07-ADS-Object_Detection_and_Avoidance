
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehiclePowertrainTractionBatteryChargingChargingPortRearRightThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight interface.
* @see Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortRearRight> testVehiclePowertrainTractionBatteryChargingChargingPortRearRight = std::make_unique<VehiclePowertrainTractionBatteryChargingChargingPortRearRightThreadSafeDecorator>(std::make_shared<VehiclePowertrainTractionBatteryChargingChargingPortRearRight>());

// Thread safe access
auto supportedInletTypes = testVehiclePowertrainTractionBatteryChargingChargingPortRearRight->getSupportedInletTypes();
testVehiclePowertrainTractionBatteryChargingChargingPortRearRight->setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum::IEC_TYPE_1_AC);
auto isFlapOpen = testVehiclePowertrainTractionBatteryChargingChargingPortRearRight->getIsFlapOpen();
testVehiclePowertrainTractionBatteryChargingChargingPortRearRight->setIsFlapOpen(false);
auto isChargingCableConnected = testVehiclePowertrainTractionBatteryChargingChargingPortRearRight->getIsChargingCableConnected();
testVehiclePowertrainTractionBatteryChargingChargingPortRearRight->setIsChargingCableConnected(false);
auto isChargingCableLocked = testVehiclePowertrainTractionBatteryChargingChargingPortRearRight->getIsChargingCableLocked();
testVehiclePowertrainTractionBatteryChargingChargingPortRearRight->setIsChargingCableLocked(false);
\endcode
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingChargingPortRearRightThreadSafeDecorator : public IVehiclePowertrainTractionBatteryChargingChargingPortRearRight
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight object to make thread safe.
    */
    explicit VehiclePowertrainTractionBatteryChargingChargingPortRearRightThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortRearRight> impl);

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight implementation. */
    void setSupportedInletTypes(Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum SupportedInletTypes) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight implementation. */
    Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum getSupportedInletTypes() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight implementation. */
    void setIsFlapOpen(bool IsFlapOpen) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight implementation. */
    bool getIsFlapOpen() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight implementation. */
    void setIsChargingCableConnected(bool IsChargingCableConnected) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight implementation. */
    bool getIsChargingCableConnected() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight implementation. */
    void setIsChargingCableLocked(bool IsChargingCableLocked) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight implementation. */
    bool getIsChargingCableLocked() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight.
    */
    IVehiclePowertrainTractionBatteryChargingChargingPortRearRightPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight object which is guarded */
    std::shared_ptr<IVehiclePowertrainTractionBatteryChargingChargingPortRearRight> m_impl;
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
