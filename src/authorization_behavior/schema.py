from pydantic import BaseModel
from typing import Literal


DecisionLabel = Literal["ALLOW", "DENY", "ESCALATE"]


class Scenario(BaseModel):
    id: str
    family: str
    variant: str = "original"

    domain: str | None = None
    environment: str | None = None
    action_type: str | None = None
    risk_level: str | None = None

    policy: str
    user_role: str
    requested_action: str
    ground_truth: DecisionLabel