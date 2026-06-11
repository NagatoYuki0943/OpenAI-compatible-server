from __future__ import annotations

import uvicorn

from openai_compatible.config import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        "openai_compatible.api:create_app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        factory=True,
    )


if __name__ == "__main__":
    main()
