"""
Phase 1 Tests — Project Setup & Dependencies
Run: .agentic-ai-final-project-env/Scripts/pytest tests/test_phase1_setup.py -v
"""

import sys
import os
import importlib
import pytest
from dotenv import load_dotenv

load_dotenv()


# ── Python version ────────────────────────────────────────────────────────────

def test_python_version():
    assert sys.version_info >= (3, 10), (
        f"Python 3.10+ required, found {sys.version_info.major}.{sys.version_info.minor}"
    )


# ── Package imports ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("module,pip_name", [
    ("langgraph",         "langgraph"),
    ("langchain_openai",  "langchain-openai"),
    ("openai",            "openai"),
    ("anthropic",         "anthropic"),
    ("numpy",             "numpy"),
    ("dotenv",            "python-dotenv"),
    ("yaml",              "pyyaml"),
    ("matplotlib",        "matplotlib"),
    ("seaborn",           "seaborn"),
    ("scipy",             "scipy"),
    ("pytest",            "pytest"),
])
def test_package_importable(module, pip_name):
    try:
        importlib.import_module(module)
    except ImportError:
        pytest.fail(f"Cannot import '{module}' — run: pip install {pip_name}")


# ── Folder structure ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("directory", [
    "src",
    "src/environment",
    "src/planners",
    "src/agent",
    "src/agent/nodes",
    "src/memory",
    "experiments",
    "experiments/configs",
    "results",
    "results/logs",
    "results/figures",
    "paper",
    "tests",
])
def test_directory_exists(directory):
    assert os.path.isdir(directory), f"Directory '{directory}/' is missing"


# ── Required files ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("filepath", [
    "requirements.txt",
    ".env.example",
    ".env",
    "PLAN.md",
    "src/__init__.py",
    "src/environment/__init__.py",
    "src/planners/__init__.py",
    "src/agent/__init__.py",
    "src/agent/nodes/__init__.py",
    "src/memory/__init__.py",
])
def test_file_exists(filepath):
    assert os.path.isfile(filepath), f"File '{filepath}' is missing"


# ── Environment / API keys ────────────────────────────────────────────────────

def test_env_file_has_llm_provider():
    provider = os.getenv("LLM_PROVIDER", "")
    assert provider in ("openai", "anthropic"), (
        f"LLM_PROVIDER must be 'openai' or 'anthropic', got '{provider}'"
    )

def test_env_file_has_llm_model():
    model = os.getenv("LLM_MODEL", "")
    assert model, "LLM_MODEL is not set in .env"

def test_openai_api_key_set():
    key = os.getenv("OPENAI_API_KEY", "")
    assert key and key != "your-openai-api-key-here", (
        "OPENAI_API_KEY is not set — add your key to .env"
    )
