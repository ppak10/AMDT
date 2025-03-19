from .base import SegmenterBase
from .gcode import SegmenterGCode

class Segmenter(SegmenterBase, SegmenterGCode):
    def __init__(self, verbose = False, **kwargs):
        """
        @param verbose: For debugging
        """
        super().__init__(verbose=verbose, **kwargs)
