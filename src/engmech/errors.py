"""Exception types raised by engmech."""

from __future__ import annotations


class EngmechError(Exception):
    """Base class for all engmech errors."""


class InputError(EngmechError, ValueError):
    """Invalid user input: bad units, unknown names, malformed geometry.

    ``where`` is a dotted path to the offending input (e.g.
    ``supports.A.normal``) and is prefixed to the message when set.
    """

    def __init__(self, message: str, where: str | None = None):
        self.message = message
        self.where = where
        # messages that already name their location are not prefixed again
        prefixed = where and not message.startswith(where)
        super().__init__(f"{where}: {message}" if prefixed else message)

    def at(self, where: str) -> InputError:
        """Return a copy located at ``where`` (outermost location wins)."""
        if self.where:
            return self
        return InputError(self.message, where)
