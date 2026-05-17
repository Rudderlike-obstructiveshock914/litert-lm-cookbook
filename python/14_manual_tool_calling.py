"""
Example 14: Manual tool calling
Set automatic_tool_calling=False so the model returns tool call
specifications instead of executing them. The application inspects each
call, decides whether to run it, then passes the result back.
"""

import json
from typing import Any

import litert_lm

MODEL_PATH = "gemma-4-E4B-it.litertlm"

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)


def _clean(value: str) -> str:
    """Strip Gemma-4 quote tokens that wrap string arguments."""
    return value.replace('<|"|>', "").strip()


def get_stock_price(ticker: str) -> float:
    """Returns the current stock price for a ticker symbol.

    Args:
        ticker: The stock ticker symbol. Must be one of: AAPL, GOOG, MSFT.
    """
    prices = {"AAPL": 189.50, "GOOG": 175.20, "MSFT": 415.80}
    return prices.get(_clean(ticker).upper(), -1.0)


def _dispatch(tool_call: dict[str, Any]) -> Any:
    fn = tool_call.get("function", {})
    name = fn.get("name", "")
    args = fn.get("arguments", {})
    print(f"  [app] executing {name}({args})")
    if name == "get_stock_price":
        return get_stock_price(**args)
    raise ValueError(f"Unknown tool: {name}")


with litert_lm.Engine(MODEL_PATH) as engine:
    with engine.create_conversation(
        tools=[get_stock_price],
        automatic_tool_calling=False,
    ) as conversation:
        query = "What are the current prices for AAPL and MSFT?"
        print(f"Q: {query}\n")

        response = conversation.send_message(query)

        # Dispatch every tool call and feed results back one by one.
        # The model responds with a text answer after each result is delivered.
        answers = []
        for call in response.get("tool_calls", []):
            result = _dispatch(call)
            tool_message = {
                "role": "tool",
                "name": call["function"]["name"],
                "content": json.dumps(result),
            }
            tool_response = conversation.send_message(tool_message)
            text = tool_response.get("content", [{}])[0].get("text", "")
            if text:
                answers.append(text)

        print("\nA: " + "  ".join(answers))
