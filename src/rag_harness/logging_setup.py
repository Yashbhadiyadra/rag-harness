"""Logging configuration shared by the CLI and the API server."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Apply a consistent log format and level to the root logger.

    Call this once at process startup — either from the CLI entry point or
    from the FastAPI lifespan hook — before any module-level loggers emit.
    """
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
