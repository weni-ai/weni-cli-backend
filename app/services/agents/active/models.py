
from pydantic import BaseModel


class ActiveAgentResourceModel(BaseModel):
    preprocessor: bytes
    rules: dict[str, bytes]
    preprocessor_example: bytes | None


class ActiveAgentPreProcessor(BaseModel):
    content: bytes
    module: str
    class_name: str


class ActiveAgentRule(BaseModel):
    content: bytes
    module_name: str