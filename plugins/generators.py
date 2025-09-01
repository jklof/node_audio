import numpy as np
from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE, DEFAULT_CHANNELS


class SineSourceNode(Node):
    NODE_TYPE = "Sine Source"
    CATEGORY = "Generators"
    DESCRIPTION = "Generates a sine wave."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("freq", data_type=float)
        self.add_output("out", data_type=np.ndarray)
        self._phase = 0.0
        self.frequency = 440.0
        self.samplerate = DEFAULT_SAMPLERATE
        self.blocksize = DEFAULT_BLOCKSIZE
        self.channels = DEFAULT_CHANNELS

    def process(self, input_data):
        # Get frequency, prioritizing the input socket
        freq_input = input_data.get("freq")
        frequency = float(freq_input) if freq_input is not None else self.frequency

        # Calculate the phase increment per sample for the current frequency
        # phase_increment = 2 * pi * frequency / samplerate
        phase_increment = (2 * np.pi * frequency) / self.samplerate

        # Generate an array of phase values for the current block
        # It starts at self._phase and increments for each sample
        phases = self._phase + np.arange(self.blocksize) * phase_increment

        # Generate the sine wave from the phase array
        output_1d = 0.5 * np.sin(phases)

        # Update the phase for the next block.
        # The new phase is the phase of the sample *after* the last one in this block.
        # We use np.mod to wrap the phase and prevent it from growing indefinitely, which avoids potential floating-point precision issues over time.
        self._phase = np.mod(phases[-1] + phase_increment, 2 * np.pi)

        # Tile to match channel count
        output_2d = np.tile(output_1d[:, np.newaxis], (1, self.channels))

        return {"out": output_2d.astype(DEFAULT_DTYPE)}
