from pydantic import BaseModel, ConfigDict


class SBase(BaseModel):
    """Базовый класс для всех Pydantic-схем в проекте."""
    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid"
    )
