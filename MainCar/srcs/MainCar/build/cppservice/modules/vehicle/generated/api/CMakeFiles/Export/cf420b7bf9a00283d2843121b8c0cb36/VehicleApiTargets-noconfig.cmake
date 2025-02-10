#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "vehicle::vehicle-api" for configuration ""
set_property(TARGET vehicle::vehicle-api APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(vehicle::vehicle-api PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libvehicle-api.so"
  IMPORTED_SONAME_NOCONFIG "libvehicle-api.so"
  )

list(APPEND _cmake_import_check_targets vehicle::vehicle-api )
list(APPEND _cmake_import_check_files_for_vehicle::vehicle-api "${_IMPORT_PREFIX}/lib/libvehicle-api.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
