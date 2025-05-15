
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehiclepowertraintractionbatterycellvoltage.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle_Powertrain_TractionBattery_CellVoltage implementation.
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTractionBatteryCellVoltage : public IVehiclePowertrainTractionBatteryCellVoltage
{
public:
    explicit VehiclePowertrainTractionBatteryCellVoltage();
    ~VehiclePowertrainTractionBatteryCellVoltage();
public:
    /**
    * Min Current voltage of the battery cell with lowest voltage.
    */
    void setMin(float Min) override;
    float getMin() const override;
    
    /**
    * IdMin Identifier of the battery cell with lowest voltage.
    */
    void setIdMin(int32_t IdMin) override;
    int32_t getIdMin() const override;
    
    /**
    * Max Current voltage of the battery cell with highest voltage.
    */
    void setMax(float Max) override;
    float getMax() const override;
    
    /**
    * IdMax Identifier of the battery cell with highest voltage.
    */
    void setIdMax(int32_t IdMax) override;
    int32_t getIdMax() const override;
    
    /**
    * CellVoltages Array of cell voltages. Length or array shall correspond to number of cells in vehicle.
    */
    void setCellVoltages(const std::list<float>& CellVoltages) override;
    const std::list<float>& getCellVoltages() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_TractionBattery_CellVoltage changes and signal emission.
    * @return The publisher for Vehicle_Powertrain_TractionBattery_CellVoltage.
    */
    IVehiclePowertrainTractionBatteryCellVoltagePublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle_Powertrain_TractionBattery_CellVoltage. */
    std::unique_ptr<IVehiclePowertrainTractionBatteryCellVoltagePublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle_Powertrain_TractionBattery_CellVoltage. */
    VehiclePowertrainTractionBatteryCellVoltageData m_data;
};
} // namespace Vehicle
} // namespace Cpp
