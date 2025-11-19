import os

import dotenv


def verbose_factory():
    dotenv.load_dotenv(
        dotenv.find_dotenv(usecwd=True),
        override=False,
    )
    return os.getenv("VERBOSE", "").lower() in ("true", "1")
