from pydantic import BaseModel


class Resource(BaseModel):
    content: bytes
    module_name: str
    class_name: str

class RuleResource(Resource):
    key: str
    template: str

class ActiveAgentResourceModel(BaseModel):
    preprocessor: Resource | None
    rules: list[RuleResource]
    preprocessor_example: bytes | None
