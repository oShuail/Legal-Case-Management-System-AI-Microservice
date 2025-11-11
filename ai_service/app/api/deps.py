from typing import Generator

# Placeholder: later we’ll return shared services (models, db, etc.)
def get_dummy_dep() -> Generator[str, None, None]:
    """Example dependency to show structure."""
    yield "ok"
