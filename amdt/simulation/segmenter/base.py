class SegmenterBase:
    """
    Base file for Segmenter class.
    """

    def __init__(self, verbose = False, **kwargs):
        self.verbose = verbose
        super().__init__()
