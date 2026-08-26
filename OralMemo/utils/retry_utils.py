from __future__ import annotations


def is_transient_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429} or (isinstance(status_code, int) and status_code >= 500):
        return True
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "timeout",
            "connection",
            "temporarily",
            "rate limit",
            "internal server",
            "openai_error",
            "bad_response_status_code",
            "upstream request failed",
            "message.content is empty",
        )
    )
