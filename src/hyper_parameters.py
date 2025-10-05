import pydantic

class HyperParameters(pydantic.BaseModel):
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 100