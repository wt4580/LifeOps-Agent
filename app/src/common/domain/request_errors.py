"""Known terminal request rejections, distinct from uncertain transport failures."""


class ChatRequestRejected(ValueError):
    def __init__(self, code, message, *, invalidated_proposal_id=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.invalidated_proposal_id = invalidated_proposal_id
