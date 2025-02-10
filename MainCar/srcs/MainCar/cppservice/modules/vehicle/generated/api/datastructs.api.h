#pragma once

#include <cinttypes>
#include <string>
#include <list>

#include "vehicle/generated/api/common.h"

namespace Cpp {
namespace Vehicle {

/**
 * Enumeration Vehicle_Powertrain_Transmission_Type_Value
 */
enum class Vehicle_Powertrain_Transmission_Type_ValueEnum {
    UNKNOWN = 0,
    SEQUENTIAL = 1,
    H = 2,
    AUTOMATIC = 3,
    DSG = 4,
    CVT = 5
};
CPP_VEHICLE_EXPORT Vehicle_Powertrain_Transmission_Type_ValueEnum toVehicle_Powertrain_Transmission_Type_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_Powertrain_Transmission_DriveType_Value
 */
enum class Vehicle_Powertrain_Transmission_DriveType_ValueEnum {
    UNKNOWN = 0,
    FORWARD_WHEEL_DRIVE = 1,
    REAR_WHEEL_DRIVE = 2,
    ALL_WHEEL_DRIVE = 3
};
CPP_VEHICLE_EXPORT Vehicle_Powertrain_Transmission_DriveType_ValueEnum toVehicle_Powertrain_Transmission_DriveType_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_Powertrain_Transmission_PerformanceMode_Value
 */
enum class Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum {
    NORMAL = 0,
    SPORT = 1,
    ECONOMY = 2,
    SNOW = 3,
    RAIN = 4
};
CPP_VEHICLE_EXPORT Vehicle_Powertrain_Transmission_PerformanceMode_ValueEnum toVehicle_Powertrain_Transmission_PerformanceMode_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_Powertrain_Transmission_GearChangeMode_Value
 */
enum class Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum {
    MANUAL = 0,
    AUTOMATIC = 1
};
CPP_VEHICLE_EXPORT Vehicle_Powertrain_Transmission_GearChangeMode_ValueEnum toVehicle_Powertrain_Transmission_GearChangeMode_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_Value
 */
enum class Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum {
    CRITICALLY_LOW = 0,
    LOW = 1,
    NORMAL = 2
};
CPP_VEHICLE_EXPORT Vehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum toVehicle_Powertrain_ElectricMotor_EngineCoolant_Level_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_Value
 */
enum class Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum {
    IEC_TYPE_1_AC = 0,
    IEC_TYPE_2_AC = 1,
    IEC_TYPE_3_AC = 2,
    IEC_TYPE_4_DC = 3,
    IEC_TYPE_1_CCS_DC = 4,
    IEC_TYPE_2_CCS_DC = 5,
    TESLA_ROADSTER = 6,
    TESLA_HPWC = 7,
    TESLA_SUPERCHARGER = 8,
    GBT_AC = 9,
    GBT_DC = 10,
    OTHER = 11
};
CPP_VEHICLE_EXPORT Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum toVehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontLeft_SupportedInletTypes_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_Value
 */
enum class Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum {
    IEC_TYPE_1_AC = 0,
    IEC_TYPE_2_AC = 1,
    IEC_TYPE_3_AC = 2,
    IEC_TYPE_4_DC = 3,
    IEC_TYPE_1_CCS_DC = 4,
    IEC_TYPE_2_CCS_DC = 5,
    TESLA_ROADSTER = 6,
    TESLA_HPWC = 7,
    TESLA_SUPERCHARGER = 8,
    GBT_AC = 9,
    GBT_DC = 10,
    OTHER = 11
};
CPP_VEHICLE_EXPORT Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum toVehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontMiddle_SupportedInletTypes_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_Value
 */
enum class Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum {
    IEC_TYPE_1_AC = 0,
    IEC_TYPE_2_AC = 1,
    IEC_TYPE_3_AC = 2,
    IEC_TYPE_4_DC = 3,
    IEC_TYPE_1_CCS_DC = 4,
    IEC_TYPE_2_CCS_DC = 5,
    TESLA_ROADSTER = 6,
    TESLA_HPWC = 7,
    TESLA_SUPERCHARGER = 8,
    GBT_AC = 9,
    GBT_DC = 10,
    OTHER = 11
};
CPP_VEHICLE_EXPORT Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum toVehicle_Powertrain_TractionBattery_Charging_ChargingPort_FrontRight_SupportedInletTypes_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_Value
 */
enum class Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum {
    IEC_TYPE_1_AC = 0,
    IEC_TYPE_2_AC = 1,
    IEC_TYPE_3_AC = 2,
    IEC_TYPE_4_DC = 3,
    IEC_TYPE_1_CCS_DC = 4,
    IEC_TYPE_2_CCS_DC = 5,
    TESLA_ROADSTER = 6,
    TESLA_HPWC = 7,
    TESLA_SUPERCHARGER = 8,
    GBT_AC = 9,
    GBT_DC = 10,
    OTHER = 11
};
CPP_VEHICLE_EXPORT Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum toVehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearLeft_SupportedInletTypes_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_Value
 */
enum class Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum {
    IEC_TYPE_1_AC = 0,
    IEC_TYPE_2_AC = 1,
    IEC_TYPE_3_AC = 2,
    IEC_TYPE_4_DC = 3,
    IEC_TYPE_1_CCS_DC = 4,
    IEC_TYPE_2_CCS_DC = 5,
    TESLA_ROADSTER = 6,
    TESLA_HPWC = 7,
    TESLA_SUPERCHARGER = 8,
    GBT_AC = 9,
    GBT_DC = 10,
    OTHER = 11
};
CPP_VEHICLE_EXPORT Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum toVehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearMiddle_SupportedInletTypes_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_Value
 */
enum class Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum {
    IEC_TYPE_1_AC = 0,
    IEC_TYPE_2_AC = 1,
    IEC_TYPE_3_AC = 2,
    IEC_TYPE_4_DC = 3,
    IEC_TYPE_1_CCS_DC = 4,
    IEC_TYPE_2_CCS_DC = 5,
    TESLA_ROADSTER = 6,
    TESLA_HPWC = 7,
    TESLA_SUPERCHARGER = 8,
    GBT_AC = 9,
    GBT_DC = 10,
    OTHER = 11
};
CPP_VEHICLE_EXPORT Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum toVehicle_Powertrain_TractionBattery_Charging_ChargingPort_RearRight_SupportedInletTypes_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_Value
 */
enum class Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum {
    IEC_TYPE_1_AC = 0,
    IEC_TYPE_2_AC = 1,
    IEC_TYPE_3_AC = 2,
    IEC_TYPE_4_DC = 3,
    IEC_TYPE_1_CCS_DC = 4,
    IEC_TYPE_2_CCS_DC = 5,
    TESLA_ROADSTER = 6,
    TESLA_HPWC = 7,
    TESLA_SUPERCHARGER = 8,
    GBT_AC = 9,
    GBT_DC = 10,
    OTHER = 11
};
CPP_VEHICLE_EXPORT Vehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum toVehicle_Powertrain_TractionBattery_Charging_ChargingPort_AnyPosition_SupportedInletTypes_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_Value
 */
enum class Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum {
    START = 0,
    STOP = 1
};
CPP_VEHICLE_EXPORT Vehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum toVehicle_Powertrain_TractionBattery_Charging_StartStopCharging_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_Value
 */
enum class Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum {
    INACTIVE = 0,
    START_TIME = 1,
    END_TIME = 2
};
CPP_VEHICLE_EXPORT Vehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum toVehicle_Powertrain_TractionBattery_Charging_Timer_Mode_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_Value
 */
enum class Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum {
    INACTIVE = 0,
    FAST_CHARGING_PREPARATION = 1,
    DRIVING_PREPARATION = 2
};
CPP_VEHICLE_EXPORT Vehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum toVehicle_Powertrain_TractionBattery_BatteryConditioning_RequestedMode_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_Powertrain_Type_Value
 */
enum class Vehicle_Powertrain_Type_ValueEnum {
    COMBUSTION = 0,
    HYBRID = 1,
    ELECTRIC = 2
};
CPP_VEHICLE_EXPORT Vehicle_Powertrain_Type_ValueEnum toVehicle_Powertrain_Type_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_Body_Lights_LightSwitch_Value
 */
enum class Vehicle_Body_Lights_LightSwitch_ValueEnum {
    OFF = 0,
    POSITION = 1,
    DAYTIME_RUNNING_LIGHTS = 2,
    AUTO = 3,
    BEAM = 4
};
CPP_VEHICLE_EXPORT Vehicle_Body_Lights_LightSwitch_ValueEnum toVehicle_Body_Lights_LightSwitch_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_Body_Lights_Brake_IsActive_Value
 */
enum class Vehicle_Body_Lights_Brake_IsActive_ValueEnum {
    INACTIVE = 0,
    ACTIVE = 1,
    ADAPTIVE = 2
};
CPP_VEHICLE_EXPORT Vehicle_Body_Lights_Brake_IsActive_ValueEnum toVehicle_Body_Lights_Brake_IsActive_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_LowVoltageSystemState_Value
 */
enum class Vehicle_LowVoltageSystemState_ValueEnum {
    UNDEFINED = 0,
    LOCK = 1,
    OFF = 2,
    ACC = 3,
    ON = 4,
    START = 5
};
CPP_VEHICLE_EXPORT Vehicle_LowVoltageSystemState_ValueEnum toVehicle_LowVoltageSystemState_ValueEnum(std::uint8_t v, bool *ok);

/**
 * Enumeration Vehicle_CurrentLocation_GNSSReceiver_FixType_Value
 */
enum class Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum {
    NONE = 0,
    TWO_D = 1,
    TWO_D_SATELLITE_BASED_AUGMENTATION = 2,
    TWO_D_GROUND_BASED_AUGMENTATION = 3,
    TWO_D_SATELLITE_AND_GROUND_BASED_AUGMENTATION = 4,
    THREE_D = 5,
    THREE_D_SATELLITE_BASED_AUGMENTATION = 6,
    THREE_D_GROUND_BASED_AUGMENTATION = 7,
    THREE_D_SATELLITE_AND_GROUND_BASED_AUGMENTATION = 8
};
CPP_VEHICLE_EXPORT Vehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum toVehicle_CurrentLocation_GNSSReceiver_FixType_ValueEnum(std::uint8_t v, bool *ok);
} // namespace Vehicle
} // namespace Cpp
