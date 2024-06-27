from .gcode_segmenter import GCodeSegmenter
from .eagar_tsai import EagarTsai

class Simulator(GCodeSegmenter, EagarTsai):
    """
    Class for creating simulations.
    """

    def __init__(self):
        super(Simulator, self).__init__()

