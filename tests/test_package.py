"""Check that the local package can be imported after environment setup."""

import llm_reasoning


def test_package_import() -> None:
    assert llm_reasoning.__version__ == "0.1.0"
