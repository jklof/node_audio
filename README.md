# node_audio
Node based realtime audio processing

# Create the complete development environment with a single command
conda env create -f environment.yml

# Activate the new environment
conda activate node-audio-dev

# remove
conda deactivate node-audio-dev
conda env remove --name node-audio-dev

# This command does two magic things:
# 1. Triggers scikit-build-core, which runs CMake to compile your C++ code.
# 2. Installs your Python package in a way that any changes to the .py
#    files are immediately reflected without needing to reinstall.
pip install -e .

