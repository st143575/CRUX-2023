from pydantic import BaseModel, ConfigDict, StringConstraints
from typing import Annotated

class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

# A string that is not empty after surrounding whitespace is removed.
_NON_EMPTY = StringConstraints(strip_whitespace=True, min_length=1)
NonEmptyStr = Annotated[str, _NON_EMPTY]