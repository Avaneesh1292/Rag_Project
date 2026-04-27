import os
import warnings

# ── Safe `observe` decorator ─────────────────────────────────────────
# Tries each known langfuse import location; falls back to a no-op so
# the rest of the app always works regardless of installed version.
try:
    from langfuse import observe          # langfuse v3+
except ImportError:
    try:
        from langfuse.decorators import observe   # langfuse v2
    except ImportError:
        warnings.warn("langfuse not installed. @observe will be a no-op.")

        def observe(name=None, **kwargs):   # type: ignore[misc]
            """Transparent no-op when langfuse is unavailable."""
            def decorator(fn):
                return fn
            return decorator if name is None else decorator


# ── Langfuse client (optional) ───────────────────────────────────────
langfuse = None

_public = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
_secret = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
_host   = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()

if _public and _secret:
    try:
        from langfuse import Langfuse
        langfuse = Langfuse(public_key=_public, secret_key=_secret, host=_host)
    except Exception as e:
        warnings.warn(f"Langfuse client init failed ({e}). Observability disabled.")
else:
    warnings.warn(
        "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set. Observability disabled.",
        stacklevel=1,
    )
