
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehiclePowertrainTractionBatteryDcdcThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Powertrain_TractionBattery_DCDC interface.
* @see Vehicle_Powertrain_TractionBattery_DCDC
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehiclePowertrainTractionBatteryDcdc> testVehiclePowertrainTractionBatteryDcdc = std::make_unique<VehiclePowertrainTractionBatteryDcdcThreadSafeDecorator>(std::make_shared<VehiclePowertrainTractionBatteryDcdc>());

// Thread safe access
auto powerLoss = testVehiclePowertrainTractionBatteryDcdc->getPowerLoss();
testVehiclePowertrainTractionBatteryDcdc->setPowerLoss(0.0f);
auto temperature = testVehiclePowertrainTractionBatteryDcdc->getTemperature();
testVehiclePowertrainTractionBatteryDcdc->setTemperature(0.0f);
\endcode
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryDcdcThreadSafeDecorator : public IVehiclePowertrainTractionBatteryDcdc
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Powertrain_TractionBattery_DCDC object to make thread safe.
    */
    explicit VehiclePowertrainTractionBatteryDcdcThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryDcdc> impl);

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_DCDC implementation. */
    void setPowerLoss(float PowerLoss) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_DCDC implementation. */
    float getPowerLoss() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_DCDC implementation. */
    void setTemperature(float Temperature) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_DCDC implementation. */
    float getTemperature() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_DCDC changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_DCDC.
    */
    IVehiclePowertrainTractionBatteryDcdcPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Powertrain_TractionBattery_DCDC object which is guarded */
    std::shared_ptr<IVehiclePowertrainTractionBatteryDcdc> m_impl;
    // Mutex for powerLoss property
    mutable std::shared_timed_mutex m_powerLossMutex;
    // Mutex for temperature property
    mutable std::shared_timed_mutex m_temperatureMutex;
};
} // namespace Vehicle
} // namespace Cpp
