"""
Smoke test: verify requirements.txt contains MetaTrader5 and does NOT
contain removed Solana / dashboard / async-DB dependencies.
Requirements: 1.3, 12.1–12.6
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"

# Read once at module level so every test shares the same content
_requirements_text = REQUIREMENTS_FILE.read_text(encoding="utf-8").lower()


def _pkg_present(name: str) -> bool:
    """Return True if `name` appears as a package name in requirements.txt."""
    return name.lower() in _requirements_text


# ── Required packages ───────────────────────────────────────────────────────

def test_metatrader5_present():
    """MetaTrader5 must be listed (Requirement 12.6)."""
    assert _pkg_present("metatrader5"), (
        "MetaTrader5 is missing from requirements.txt. "
        "Add 'MetaTrader5>=5.0.45'."
    )


# ── Removed dashboard packages ───────────────────────────────────────────────

def test_streamlit_removed():
    """streamlit must not be listed (Requirement 12.2)."""
    assert not _pkg_present("streamlit"), (
        "streamlit is still present in requirements.txt but must be removed."
    )


def test_plotly_removed():
    """plotly must not be listed (Requirement 12.2)."""
    assert not _pkg_present("plotly"), (
        "plotly is still present in requirements.txt but must be removed."
    )


# ── Removed Solana packages ──────────────────────────────────────────────────

def test_solders_removed():
    """solders must not be listed (Requirement 12.1)."""
    assert not _pkg_present("solders"), (
        "solders is still present in requirements.txt but must be removed."
    )


def test_solana_removed():
    """solana must not be listed (Requirement 12.1)."""
    assert not _pkg_present("solana"), (
        "solana is still present in requirements.txt but must be removed."
    )


def test_base58_removed():
    """base58 must not be listed (Requirement 12.1)."""
    assert not _pkg_present("base58"), (
        "base58 is still present in requirements.txt but must be removed."
    )


# ── Removed async / DB packages ──────────────────────────────────────────────

def test_aiohttp_removed():
    """aiohttp must not be listed (Requirement 12.3)."""
    assert not _pkg_present("aiohttp"), (
        "aiohttp is still present in requirements.txt but must be removed."
    )


def test_asyncpg_removed():
    """asyncpg must not be listed (Requirement 12.4)."""
    assert not _pkg_present("asyncpg"), (
        "asyncpg is still present in requirements.txt but must be removed."
    )


def test_psycopg2_removed():
    """psycopg2-binary must not be listed (Requirement 12.4)."""
    assert not _pkg_present("psycopg2"), (
        "psycopg2-binary is still present in requirements.txt but must be removed."
    )


def test_sqlalchemy_removed():
    """sqlalchemy must not be listed (Requirement 12.5)."""
    assert not _pkg_present("sqlalchemy"), (
        "sqlalchemy is still present in requirements.txt but must be removed."
    )
