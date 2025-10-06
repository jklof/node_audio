cmake_minimum_required(VERSION 3.15)
project(gain_processor CXX)

# --- Target Definition ---
add_library(gain_processor SHARED gain_processor.cpp)

set_target_properties(gain_processor PROPERTIES
    # General output directory for the DLL/shared library
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../plugins"
    
    # Explicitly set the output directory for the RELEASE configuration
    # This is the key that overrides MSBuild's default of adding "/Release"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/../plugins"
)

