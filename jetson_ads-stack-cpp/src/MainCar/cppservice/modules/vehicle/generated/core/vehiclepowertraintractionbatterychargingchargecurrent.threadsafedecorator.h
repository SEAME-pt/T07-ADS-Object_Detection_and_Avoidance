
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehiclePowertrainTractionBatteryChargingChargeCurrentThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent interface.
* @see Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehiclePowertrainTractionBatteryChargingChargeCurrent> testVehiclePowertrainTractionBatteryChargingChargeCurrent = std::make_unique<VehiclePowertrainTractionBatteryChargingChargeCurrentThreadSafeDecorator>(std::make_shared<VehiclePowertrainTractionBatteryChargingChargeCurrent>());

// Thread safe access
auto dc = testVehiclePowertrainTractionBatteryChargingChargeCurrent->getDc();
testVehiclePowertrainTractionBatteryChargingChargeCurrent->setDc(0.0f);
auto phase1 = testVehiclePowertrainTractionBatteryChargingChargeCurrent->getPhase1();
testVehiclePowertrainTractionBatteryChargingChargeCurrent->setPhase1(0.0f);
auto phase2 = testVehiclePowertrainTractionBatteryChargingChargeCurrent->getPhase2();
testVehiclePowertrainTractionBatteryChargingChargeCurrent->setPhase2(0.0f);
auto phase3 = testVehiclePowertrainTractionBatteryChargingChargeCurrent->getPhase3();
testVehiclePowertrainTractionBatteryChargingChargeCurrent->setPhase3(0.0f);
\endcode
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryChargingChargeCurrentThreadSafeDecorator : public IVehiclePowertrainTractionBatteryChargingChargeCurrent
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent object to make thread safe.
    */
    explicit VehiclePowertrainTractionBatteryChargingChargeCurrentThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryChargingChargeCurrent> impl);

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent implementation. */
    void setDc(float DC) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent implementation. */
    float getDc() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent implementation. */
    void setPhase1(float Phase1) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent implementation. */
    float getPhase1() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent implementation. */
    void setPhase2(float Phase2) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent implementation. */
    float getPhase2() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent implementation. */
    void setPhase3(float Phase3) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent implementation. */
    float getPhase3() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent.
    */
    IVehiclePowertrainTractionBatteryChargingChargeCurrentPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Powertrain_TractionBattery_Charging_ChargeCurrent object which is guarded */
    std::shared_ptr<IVehiclePowertrainTractionBatteryChargingChargeCurrent> m_impl;
    // Mutex for dc property
    mutable std::shared_timed_mutex m_dcMutex;
    // Mutex for phase1 property
    mutable std::shared_timed_mutex m_phase1Mutex;
    // Mutex for phase2 property
    mutable std::shared_timed_mutex m_phase2Mutex;
    // Mutex for phase3 property
    mutable std::shared_timed_mutex m_phase3Mutex;
};
} // namespace Vehicle
} // namespace Cpp
