from pydantic import BaseModel
from typing import Literal


DecisionLabel = Literal["ALLOW", "DENY", "ESCALATE"]


class Scenario(BaseModel):
    id: str
    policy: str
    user_role: str
    requested_action: str
    ground_truth: DecisionLabel