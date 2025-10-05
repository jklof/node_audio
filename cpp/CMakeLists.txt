cmake_minimum_required(VERSION 3.15)
project(gain_processor CXX)

# Create the shared library (.dll/.so)
add_library(gain_processor SHARED gain_processor.cpp)

# --- CRITICAL PART ---
# Tell CMake where to put the final .dll or .so file.
# We want it right next to the Python plugin that will use it.
set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/../plugins")