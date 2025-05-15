
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehiclePowertrainTractionBatteryTemperatureThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Powertrain_TractionBattery_Temperature interface.
* @see Vehicle_Powertrain_TractionBattery_Temperature
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehiclePowertrainTractionBatteryTemperature> testVehiclePowertrainTractionBatteryTemperature = std::make_unique<VehiclePowertrainTractionBatteryTemperatureThreadSafeDecorator>(std::make_shared<VehiclePowertrainTractionBatteryTemperature>());

// Thread safe access
auto average = testVehiclePowertrainTractionBatteryTemperature->getAverage();
testVehiclePowertrainTractionBatteryTemperature->setAverage(0.0f);
auto min = testVehiclePowertrainTractionBatteryTemperature->getMin();
testVehiclePowertrainTractionBatteryTemperature->setMin(0.0f);
auto max = testVehiclePowertrainTractionBatteryTemperature->getMax();
testVehiclePowertrainTractionBatteryTemperature->setMax(0.0f);
auto cellTemperature = testVehiclePowertrainTractionBatteryTemperature->getCellTemperature();
testVehiclePowertrainTractionBatteryTemperature->setCellTemperature(std::list<float>());
\endcode
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryTemperatureThreadSafeDecorator : public IVehiclePowertrainTractionBatteryTemperature
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Powertrain_TractionBattery_Temperature object to make thread safe.
    */
    explicit VehiclePowertrainTractionBatteryTemperatureThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryTemperature> impl);

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Temperature implementation. */
    void setAverage(float Average) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Temperature implementation. */
    float getAverage() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Temperature implementation. */
    void setMin(float Min) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Temperature implementation. */
    float getMin() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Temperature implementation. */
    void setMax(float Max) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Temperature implementation. */
    float getMax() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Temperature implementation. */
    void setCellTemperature(const std::list<float>& CellTemperature) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_Temperature implementation. */
    const std::list<float>& getCellTemperature() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_Temperature changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_Temperature.
    */
    IVehiclePowertrainTractionBatteryTemperaturePublisher& _getPublisher() const override;
private:
    /** The Vehicle_Powertrain_TractionBattery_Temperature object which is guarded */
    std::shared_ptr<IVehiclePowertrainTractionBatteryTemperature> m_impl;
    // Mutex for average property
    mutable std::shared_timed_mutex m_averageMutex;
    // Mutex for min property
    mutable std::shared_timed_mutex m_minMutex;
    // Mutex for max property
    mutable std::shared_timed_mutex m_maxMutex;
    // Mutex for cellTemperature property
    mutable std::shared_timed_mutex m_cellTemperatureMutex;
};
} // namespace Vehicle
} // namespace Cpp
