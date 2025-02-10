
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include "vehicle/generated/core/vehicle.data.h"
#include <memory>

namespace Cpp {
namespace Vehicle {

/**
* The Vehicle implementation.
*/
class CPP_VEHICLE_EXPORT Vehicle : public IVehicle
{
public:
    explicit Vehicle();
    ~Vehicle();
public:
    /**
    * LowVoltageSystemState State of the supply voltage of the control units (usually 12V).
    */
    void setLowVoltageSystemState(Vehicle_LowVoltageSystemState_ValueEnum LowVoltageSystemState) override;
    Vehicle_LowVoltageSystemState_ValueEnum getLowVoltageSystemState() const override;
    
    /**
    * Speed Vehicle speed.
    */
    void setSpeed(float Speed) override;
    float getSpeed() const override;
    
    /**
    * TraveledDistance Odometer reading, total distance traveled during the lifetime of the vehicle.
    */
    void setTraveledDistance(float TraveledDistance) override;
    float getTraveledDistance() const override;
    
    /**
    * TraveledDistanceSinceStart Distance traveled since start of current trip.
    */
    void setTraveledDistanceSinceStart(float TraveledDistanceSinceStart) override;
    float getTraveledDistanceSinceStart() const override;
    
    /**
    * StartTime Start time of current or latest trip, formatted according to ISO 8601 with UTC time zone.
    */
    void setStartTime(const std::string& StartTime) override;
    const std::string& getStartTime() const override;
    
    /**
    * TripDuration Duration of latest trip.
    */
    void setTripDuration(float TripDuration) override;
    float getTripDuration() const override;
    
    /**
    * TripMeterReading Trip meter reading.
    */
    void setTripMeterReading(float TripMeterReading) override;
    float getTripMeterReading() const override;
    
    /**
    * IsBrokenDown Vehicle breakdown or any similar event causing vehicle to stop on the road, that might pose a risk to other road users. True = Vehicle broken down on the road, due to e.g. engine problems, flat tire, out of gas, brake problems. False = Vehicle not broken down.
    */
    void setIsBrokenDown(bool IsBrokenDown) override;
    bool getIsBrokenDown() const override;
    
    /**
    * IsMoving Indicates whether the vehicle is stationary or moving.
    */
    void setIsMoving(bool IsMoving) override;
    bool getIsMoving() const override;
    
    /**
    * AverageSpeed Average speed for the current trip.
    */
    void setAverageSpeed(float AverageSpeed) override;
    float getAverageSpeed() const override;
    
    /**
    * PowerOptimizeLevel Power optimization level for this branch/subsystem. A higher number indicates more aggressive power optimization. Level 0 indicates that all functionality is enabled, no power optimization enabled. Level 10 indicates most aggressive power optimization mode, only essential functionality enabled.
    */
    void setPowerOptimizeLevel(int32_t PowerOptimizeLevel) override;
    int32_t getPowerOptimizeLevel() const override;
    
    /**
    * IsAutoPowerOptimize Auto Power Optimization Flag When set to 'true', the system enables automatic power optimization, dynamically adjusting the power optimization level based on runtime conditions or features managed by the OEM. When set to 'false', manual control of the power optimization level is allowed.
    */
    void setIsAutoPowerOptimize(bool IsAutoPowerOptimize) override;
    bool getIsAutoPowerOptimize() const override;
    
    /**
    * Access to a publisher, use it to subscribe for Vehicle changes and signal emission.
    * @return The publisher for Vehicle.
    */
    IVehiclePublisher& _getPublisher() const override;
private:
    /** The publisher for the Vehicle. */
    std::unique_ptr<IVehiclePublisher> m_publisher;
    /** The helper structure to store all the properties for Vehicle. */
    VehicleData m_data;
};
} // namespace Vehicle
} // namespace Cpp
