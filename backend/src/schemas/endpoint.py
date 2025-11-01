from typing import Optional, Dict

from pydantic import BaseModel


class HealthStatus(BaseModel):
    status: str
    details: Optional[Dict[str, str]] = None