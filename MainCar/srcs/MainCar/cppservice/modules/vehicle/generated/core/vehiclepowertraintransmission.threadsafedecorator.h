
#pragma once
#include "vehicle/generated/api/vehicle.h"
#include "vehicle/generated/api/common.h"
#include <memory>
#include <shared_mutex>

namespace Cpp {
namespace Vehicle {

/** 
* @brief The VehiclePowertrainTransmissionThreadSafeDecorator can be used to make property access thread safe.
*
* Each property is guarded with its own mutex.
* Multiple read/get operations can occur at the same time but only one write/set operation at a time.
*
* Operations are not guarded by default since the function logic can be too complex than to simply lock it.
* However, functions can be locked by just adding the same mechanism in the implementation file of
* the Vehicle_Powertrain_Transmission interface.
* @see Vehicle_Powertrain_Transmission
*
\code{.cpp}
using namespace Cpp::Vehicle;

std::unique_ptr<IVehiclePowertrainTransmission> testVehiclePowertrainTransmission = std::make_unique<VehiclePowertrainTransmissionThreadSafeDecorator>(std::make_shared<VehiclePowertrainTransmission>());

// Thread safe access
auto type = testVehiclePowertrainTransmission->getType();
testVehiclePowertrainTransmission->setType(Vehicle_Powertrain_Transmission_Type_ValueEnum::UNKNOWN);
auto gearCount = testVehiclePowertrainTransmission->getGearCount();
testVehiclePowertrainTransmission->setGearCount(0);
auto driveType = testVehiclePowertrainTransmission->getDriveType();
testVehiclePowertrainTransmission->setDriveType(Vehicle_Powertrain_Transmission_DriveType_ValueEnum::UNKNOWN);
auto travelledDistance = testVehiclePowertrainTransmission->getTravelledDistance();
testVehiclePowertrainTransmission->setTravelledDistance(0.0f);
auto currentGear = testVehiclePowertrainTransmission->getCurrentGear();
testVehiclePowertrainTransmission->setCurrentGear(0);
auto selectedGear = testVehiclePowertrainTransmission->getSelectedGear();
testVehiclePowertrainTransmission->setSelectedGear(0);
auto isParkLockEngaged = testVehiclePowertrainTransmission->getIsParkLockEngaged();
testVehiclePowertrainTransmission->setIsParkLockEngaged(false);
auto isLowRangeEngaged = testVehiclePowertrainTransmission->getIsLowRangeEngaged();
testVehiclePowertrainTransmission->setIsLowRangeEngaged(false);
auto isElectricalPowertrainEngaged = testVehiclePowertrainTransmission->getIsElectricalPowertrainEngaged();
testVehiclePowertrainTransmission->setIsElectricalPowertrainEngaged(false);
auto performanceMode = testVehiclePowertrainTransmission->getPerformanceMode();
testVehiclePowertrainTransmission->setPerformanceMode(Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum::NORMAL);
auto gearChangeMode = testVehiclePowertrainTransmission->getGearChangeMode();
testVehiclePowertrainTransmission->setGearChangeMode(Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum::MANUAL);
auto temperature = testVehiclePowertrainTransmission->getTemperature();
testVehiclePowertrainTransmission->setTemperature(0.0f);
auto clutchEngagement = testVehiclePowertrainTransmission->getClutchEngagement();
testVehiclePowertrainTransmission->setClutchEngagement(0.0f);
auto clutchWear = testVehiclePowertrainTransmission->getClutchWear();
testVehiclePowertrainTransmission->setClutchWear(0);
auto diffLockFrontEngagement = testVehiclePowertrainTransmission->getDiffLockFrontEngagement();
testVehiclePowertrainTransmission->setDiffLockFrontEngagement(0.0f);
auto diffLockRearEngagement = testVehiclePowertrainTransmission->getDiffLockRearEngagement();
testVehiclePowertrainTransmission->setDiffLockRearEngagement(0.0f);
auto torqueDistribution = testVehiclePowertrainTransmission->getTorqueDistribution();
testVehiclePowertrainTransmission->setTorqueDistribution(0.0f);
\endcode
*/
class CPP_VEHICLE_EXPORT VehiclePowertrainTransmissionThreadSafeDecorator : public IVehiclePowertrainTransmission
{
public:
    /** 
    * ctor
    * @param impl The Vehicle_Powertrain_Transmission object to make thread safe.
    */
    explicit VehiclePowertrainTransmissionThreadSafeDecorator(std::shared_ptr<IVehiclePowertrainTransmission> impl);

    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    void setType(Vehicle_Powertrain_Transmission_Type_ValueEnum Type) override;
    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    Vehicle_Powertrain_Transmission_Type_ValueEnum getType() const override;

    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    void setGearCount(int32_t GearCount) override;
    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    int32_t getGearCount() const override;

    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    void setDriveType(Vehicle_Powertrain_Transmission_DriveType_ValueEnum DriveType) override;
    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    Vehicle_Powertrain_Transmission_DriveType_ValueEnum getDriveType() const override;

    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    void setTravelledDistance(float TravelledDistance) override;
    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    float getTravelledDistance() const override;

    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    void setCurrentGear(int32_t CurrentGear) override;
    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    int32_t getCurrentGear() const override;

    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    void setSelectedGear(int32_t SelectedGear) override;
    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    int32_t getSelectedGear() const override;

    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    void setIsParkLockEngaged(bool IsParkLockEngaged) override;
    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    bool getIsParkLockEngaged() const override;

    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    void setIsLowRangeEngaged(bool IsLowRangeEngaged) override;
    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    bool getIsLowRangeEngaged() const override;

    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    void setIsElectricalPowertrainEngaged(bool IsElectricalPowertrainEngaged) override;
    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    bool getIsElectricalPowertrainEngaged() const override;

    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    void setPerformanceMode(Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum PerformanceMode) override;
    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum getPerformanceMode() const override;

    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    void setGearChangeMode(Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum GearChangeMode) override;
    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum getGearChangeMode() const override;

    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    void setTemperature(float Temperature) override;
    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    float getTemperature() const override;

    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    void setClutchEngagement(float ClutchEngagement) override;
    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    float getClutchEngagement() const override;

    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    void setClutchWear(int32_t ClutchWear) override;
    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    int32_t getClutchWear() const override;

    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    void setDiffLockFrontEngagement(float DiffLockFrontEngagement) override;
    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    float getDiffLockFrontEngagement() const override;

    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    void setDiffLockRearEngagement(float DiffLockRearEngagement) override;
    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    float getDiffLockRearEngagement() const override;

    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    void setTorqueDistribution(float TorqueDistribution) override;
    /** Guards and forwards call to Vehicle_Powertrain_Transmission implementation. */
    float getTorqueDistribution() const override;

    /**
    * Access to a publisher, use it to subscribe for Vehicle_Powertrain_Transmission changes and signal emission.
    * This call is thread safe.
    * @return The publisher for Vehicle_Powertrain_Transmission.
    */
    IVehiclePowertrainTransmissionPublisher& _getPublisher() const override;
private:
    /** The Vehicle_Powertrain_Transmission object which is guarded */
    std::shared_ptr<IVehiclePowertrainTransmission> m_impl;
    // Mutex for type property
    mutable std::shared_timed_mutex m_typeMutex;
    // Mutex for gearCount property
    mutable std::shared_timed_mutex m_gearCountMutex;
    // Mutex for driveType property
    mutable std::shared_timed_mutex m_driveTypeMutex;
    // Mutex for travelledDistance property
    mutable std::shared_timed_mutex m_travelledDistanceMutex;
    // Mutex for currentGear property
    mutable std::shared_timed_mutex m_currentGearMutex;
    // Mutex for selectedGear property
    mutable std::shared_timed_mutex m_selectedGearMutex;
    // Mutex for isParkLockEngaged property
    mutable std::shared_timed_mutex m_isParkLockEngagedMutex;
    // Mutex for isLowRangeEngaged property
    mutable std::shared_timed_mutex m_isLowRangeEngagedMutex;
    // Mutex for isElectricalPowertrainEngaged property
    mutable std::shared_timed_mutex m_isElectricalPowertrainEngagedMutex;
    // Mutex for performanceMode property
    mutable std::shared_timed_mutex m_performanceModeMutex;
    // Mutex for gearChangeMode property
    mutable std::shared_timed_mutex m_gearChangeModeMutex;
    // Mutex for temperature property
    mutable std::shared_timed_mutex m_temperatureMutex;
    // Mutex for clutchEngagement property
    mutable std::shared_timed_mutex m_clutchEngagementMutex;
    // Mutex for clutchWear property
    mutable std::shared_timed_mutex m_clutchWearMutex;
    // Mutex for diffLockFrontEngagement property
    mutable std::shared_timed_mutex m_diffLockFrontEngagementMutex;
    // Mutex for diffLockRearEngagement property
    mutable std::shared_timed_mutex m_diffLockRearEngagementMutex;
    // Mutex for torqueDistribution property
    mutable std::shared_timed_mutex m_torqueDistributionMutex;
};
} // namespace Vehicle
} // namespace Cpp
