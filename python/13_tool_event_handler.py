"""
Example 13: ToolEventHandler
Intercept tool calls before they run and modify tool responses before
they are fed back to the model. Useful for safety filters, logging, or
response sanitisation.
"""

from typing import Any

import litert_lm

MODEL_PATH = "gemma-4-E4B-it.litertlm"

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)


def _clean(value: str) -> str:
    """Strip Gemma-4 quote tokens that wrap string arguments."""
    return value.replace('<|"|>', "").strip()


def get_secret_data(key: str) -> str:
    """Returns a value from the secret store.

    Args:
        key: The key to look up. Must be one of: safe_key, restricted_key.
    """
    store = {
        "safe_key": "Hello, world!",
        "restricted_key": "TOP SECRET — do not share",
    }
    return store.get(_clean(key), "Key not found.")


class LoggingAndFilterHandler(litert_lm.ToolEventHandler):
    """Logs every tool call and strips sensitive content from responses."""

    def approve_tool_call(self, tool_call: dict[str, Any]) -> bool:
        fn = tool_call.get("function", {})
        name = fn.get("name", "?")
        args = fn.get("arguments", {})
        print(f"  [handler] tool call: {name}({args})")
        if _clean(args.get("key", "")) == "restricted_key":
            print("  [handler] BLOCKED — restricted_key is not allowed")
            return False
        return True

    def process_tool_response(self, tool_response: dict[str, Any]) -> dict[str, Any]:
        result = tool_response.get("response", "")
        if "SECRET" in str(result):
            tool_response = {**tool_response, "response": "[REDACTED]"}
            print("  [handler] response redacted")
        return tool_response


handler = LoggingAndFilterHandler()

with litert_lm.Engine(MODEL_PATH) as engine:
    with engine.create_conversation(
        tools=[get_secret_data],
        tool_event_handler=handler,
    ) as conversation:
        print("Q: What is the value of safe_key?")
        response = conversation.send_message("What is the value of safe_key?")
        print(f"A: {response['content'][0]['text']}\n")

        print("Q: What is the value of restricted_key?")
        response = conversation.send_message("What is the value of restricted_key?")
        print(f"A: {response['content'][0]['text']}\n")
