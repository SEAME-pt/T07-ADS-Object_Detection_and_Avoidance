#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "vehicle::vehicle-implementation" for configuration ""
set_property(TARGET vehicle::vehicle-implementation APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(vehicle::vehicle-implementation PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_NOCONFIG "vehicle::vehicle-core"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libvehicle-implementation.so"
  IMPORTED_SONAME_NOCONFIG "libvehicle-implementation.so"
  )

list(APPEND _cmake_import_check_targets vehicle::vehicle-implementation )
list(APPEND _cmake_import_check_files_for_vehicle::vehicle-implementation "${_IMPORT_PREFIX}/lib/libvehicle-implementation.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
