cmake_minimum_required(VERSION 3.15)
project(gain_processor CXX)

# --- Reusable function to copy target to directory ---
function(copy_target_to_directory target_name destination_dir)
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${destination_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy 
            "$<TARGET_FILE:${target_name}>" 
            "${destination_dir}/$<TARGET_FILE_NAME:${target_name}>"
        COMMENT "Moving ${target_name} to ${destination_dir}"
    )
endfunction()

# --- Target Definition ---
add_library(gain_processor SHARED gain_processor.cpp)

# Define the target directory
set(TARGET_DIR "${PROJECT_SOURCE_DIR}/../plugins")

# Use the reusable function
copy_target_to_directory(gain_processor "${TARGET_DIR}")