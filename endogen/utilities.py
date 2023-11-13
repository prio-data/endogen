from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass
class PanelUnits:
    """Utility class for describing panel-unit names.

    The user might want to use different naming conventions than we do under-the-hood. Since we use nixtla for
    parts of our tasks, we use their naming conventions ("ds" for time and "unique_id" for units).

    Parameters
    ----------
    time_var : str
        The name of the time variable as given by the user
    unit_var : str
        The name of the unit variable as given by the user
    """

    time_var: str
    unit_var: str

    def to_dict(self, inv: bool = False) -> Mapping[str, str]:
        """Return a mapping of user-provided and internal ids

        Parameters
        ----------
        inv : bool
            Invert the mapping. {internal: user} instead of {user: internal}
        """
        if inv:
            return {"ds": self.time_var, "unique_id": self.unit_var}
        else:
            return {self.time_var: "ds", self.unit_var: "unique_id"}

    @property
    def internal_index(self) -> Sequence[str]:
        """Return the index sequence. Useful to get a consistent use across code."""
        return ["unique_id", "ds"]

    @property
    def external_index(self) -> Sequence[str]:
        """Return the index sequence. Useful to get a consistent use across code."""
        return [self.unit_var, self.time_var]
