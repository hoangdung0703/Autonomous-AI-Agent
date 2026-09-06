"""calculate_or_verify(expression, context) tool.

Safely evaluates a numeric expression using asteval (NOT Python eval()).
This is the key differentiator from plain RAG: the agent computes, not just
retrieves (Section 5.5).
"""

from asteval import Interpreter
from pydantic import BaseModel, Field

aeval = Interpreter()


class CalculateOrVerifyParams(BaseModel):
    expression: str = Field(
        ...,
        description="A numeric math expression to evaluate, e.g. '(8/12) * 12'. Use only after retrieving the relevant formula or numbers from the knowledge base — never invent numbers.",
    )
    context: str = Field(
        ...,
        description="A short plain-text description of what this calculation represents, e.g. 'annual leave for 8 months worked'.",
    )


async def calculate_or_verify(expression: str, context: str) -> str:
    result = aeval(expression)
    if aeval.error:
        message = "; ".join(err.get_error()[1] for err in aeval.error)
        aeval.error = []
        raise ValueError(f"Invalid expression: {expression} ({message})")
    return f"Result: {result} (context: {context})"
