
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehiclePowertrainTractionBatteryCellVoltageThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Powertrain_TractionBattery_CellVoltage interface.
* @see Vehicle_Powertrain_TractionBattery_CellVoltage
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehiclePowertrainTractionBatteryCellVoltage> testVehiclePowertrainTractionBatteryCellVoltage = std::make_unique<VehiclePowertrainTractionBatteryCellVoltageThreadSafeDecorator>(std::make_shared<VehiclePowertrainTractionBatteryCellVoltage>());

// Thread safe access
auto min = testVehiclePowertrainTractionBatteryCellVoltage->getMin();
testVehiclePowertrainTractionBatteryCellVoltage->setMin(0.0f);
auto idMin = testVehiclePowertrainTractionBatteryCellVoltage->getIdMin();
testVehiclePowertrainTractionBatteryCellVoltage->setIdMin(0);
auto max = testVehiclePowertrainTractionBatteryCellVoltage->getMax();
testVehiclePowertrainTractionBatteryCellVoltage->setMax(0.0f);
auto idMax = testVehiclePowertrainTractionBatteryCellVoltage->getIdMax();
testVehiclePowertrainTractionBatteryCellVoltage->setIdMax(0);
auto cellVoltages = testVehiclePowertrainTractionBatteryCellVoltage->getCellVoltages();
testVehiclePowertrainTractionBatteryCellVoltage->setCellVoltages(std::list<float>());
\endcode
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryCellVoltageThreadSafeDecorator : public IVehiclePowertrainTractionBatteryCellVoltage
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Powertrain_TractionBattery_CellVoltage object to make thread safe.
    */
    explicit VehiclePowertrainTractionBatteryCellVoltageThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTractionBatteryCellVoltage> impl);

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_CellVoltage implementation. */
    void setMin(float Min) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_CellVoltage implementation. */
    float getMin() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_CellVoltage implementation. */
    void setIdMin(int32_t IdMin) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_CellVoltage implementation. */
    int32_t getIdMin() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_CellVoltage implementation. */
    void setMax(float Max) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_CellVoltage implementation. */
    float getMax() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_CellVoltage implementation. */
    void setIdMax(int32_t IdMax) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_CellVoltage implementation. */
    int32_t getIdMax() const override;

    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_CellVoltage implementation. */
    void setCellVoltages(const std::list<float>& CellVoltages) override;
    /** Guards and forwards call to Vehicle_Powertrain_TractionBattery_CellVoltage implementation. */
    const std::list<float>& getCellVoltages() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_CellVoltage changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_CellVoltage.
    */
    IVehiclePowertrainTractionBatteryCellVoltagePublisher& _getPublisher() const override;
private:
    /** The Vehicle_Powertrain_TractionBattery_CellVoltage object which is guarded */
    std::shared_ptr<IVehiclePowertrainTractionBatteryCellVoltage> m_impl;
    // Mutex for min property
    mutable std::shared_timed_mutex m_minMutex;
    // Mutex for idMin property
    mutable std::shared_timed_mutex m_idMinMutex;
    // Mutex for max property
    mutable std::shared_timed_mutex m_maxMutex;
    // Mutex for idMax property
    mutable std::shared_timed_mutex m_idMaxMutex;
    // Mutex for cellVoltages property
    mutable std::shared_timed_mutex m_cellVoltagesMutex;
};
} // namespace Vehicle
} // namespace Cpp
