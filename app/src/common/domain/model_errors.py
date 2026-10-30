"""Safe application-level failures for unusable provider completions."""


class ModelOutputIncompleteError(RuntimeError):
    def __init__(self):
        super().__init__("Model output did not complete")


class ModelOutputInvalidError(RuntimeError):
    def __init__(self):
        super().__init__("Model output did not match the required structure")
