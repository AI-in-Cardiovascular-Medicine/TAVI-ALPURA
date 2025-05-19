from collections import defaultdict


class NestedDefaultDict(defaultdict):
    """Nested dict, which can be dynamically expanded"""

    def __init__(self, _=None):
        super().__init__(self.__class__)

    def __repr__(self):
        return repr(dict(self))
