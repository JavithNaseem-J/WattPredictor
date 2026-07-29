class CustomException(Exception):
    """Legacy exception wrapper. Prefers standard Python exceptions with logger.exception()."""
    def __init__(self, message, error_detail=None):
        super().__init__(str(message))
        self.message = str(message)
