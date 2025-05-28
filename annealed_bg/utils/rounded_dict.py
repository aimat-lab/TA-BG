class RoundedKeyDict(dict):
    def __init__(self, digits: int = 2, *args, **kwargs):
        """
        A dictionary that rounds keys to a specified number of digits.

        Args:
            digits: Number of digits to round keys to.
        """

        self.digits = digits
        super().__init__(*args, **kwargs)

    def _round_key(self, key):
        if isinstance(key, (float, int)):
            return round(float(key), self.digits)
        raise TypeError("Keys must be of type float or int")

    def __setitem__(self, key, value):
        rounded_key = self._round_key(key)
        super().__setitem__(rounded_key, value)

    def __getitem__(self, key):
        rounded_key = self._round_key(key)
        return super().__getitem__(rounded_key)

    def __delitem__(self, key):
        rounded_key = self._round_key(key)
        super().__delitem__(rounded_key)

    def __contains__(self, key):
        rounded_key = self._round_key(key)
        return super().__contains__(rounded_key)

    def to_dict(self):
        return dict(self)

    def fill_with_dict(self, d):
        for k, v in d.items():
            self[k] = v
