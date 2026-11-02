"""Hosted-mode isolation and capability checks.

Run with the slim install:  python tests/test_hosted_mode.py
Not a pytest file; plain asserts so it runs without extra deps.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import tempfile
TMP = tempfile.mkdtemp(prefix="edit-helper-test-")


def use_temp_db():
    """Point the app at a throwaway DB before gui.app is re-imported.

    db.database reads the module-global DB_PATH when the Database singleton is
    created, so patch it there - patching config.settings alone would not move
    the database. The real data/edit_helper.db of whatever checkout these
    tests run in is never touched.
    """
    for m in ("gui.app",):
        sys.modules.pop(m, None)
    import db.database as dbm
    dbm.DB_PATH = Path(TMP) / "data" / "edit_helper.db"
    dbm.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    dbm._db_instance = None
    import config.settings as cs
    cs.settings.TEMP_DIR = Path(TMP) / "temp_cache"
    cs.settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)


def load_app(hosted: bool, env_extra: dict | None = None):
    """(Re)load gui.app with a clean hosted-mode environment."""
    for k in ("EDIT_HELPER_HOSTED", "EDIT_HELPER_ACCESS_CODE"):
        os.environ.pop(k, None)
    if env_extra:
        for k, v in env_extra.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    if hosted:
        os.environ["EDIT_HELPER_HOSTED"] = "1"
    # fresh settings too: earlier tests monkeypatch key fields on the singleton
    sys.modules.pop("config.settings", None)
    use_temp_db()
    import gui.app as app
    return app


class _State(dict):
    """Dict with attribute access, standing in for st.session_state in tests."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def set_state(mapping: dict):
    st.session_state = _State(mapping)  # noqa: bare-mode test process only


def main():
    # ── key isolation ────────────────────────────────────────
    app = load_app(hosted=True)
    set_state({})
    app.db.set_setting("gemini_api_key", "SAVED-DB-KEY")
    app.db.set_setting("pexels_api_key", "SAVED-DB-PEXELS")
    try:
        assert app.get_gemini_key() == "", "hosted must ignore DB-saved Gemini key"
        assert app.get_pexels_key() == "", "hosted must ignore DB-saved Pexels key"
        print("PASS hosted: DB-saved keys are never read")

        set_state({"gemini_api_key_session": "SESS-KEY", "pexels_api_key_session": "SESS-PEX"})
        assert app.get_gemini_key() == "SESS-KEY"
        assert app.get_pexels_key() == "SESS-PEX"
        print("PASS hosted: session keys win")

        set_state({})
        app.settings.GEMINI_API_KEY = "HOST-ENV-KEY"
        app.settings.PEXELS_API_KEY = "HOST-ENV-PEX"
        assert app.get_gemini_key() == "HOST-ENV-KEY"
        assert app.get_pexels_key() == "HOST-ENV-PEX"
        print("PASS hosted: host env/settings fallback works (host-controlled, not DB)")
    finally:
        app.db.set_setting("gemini_api_key", "")
        app.db.set_setting("pexels_api_key", "")

    # non-hosted keeps old behaviour (DB allowed)
    app = load_app(hosted=False)
    set_state({})
    app.db.set_setting("gemini_api_key", "SAVED-DB-KEY")
    try:
        assert app.get_gemini_key() == "SAVED-DB-KEY", "local mode should read DB key"
        print("PASS local: DB-saved key still works")
    finally:
        app.db.set_setting("gemini_api_key", "")

    # ── capability flags ─────────────────────────────────────
    app = load_app(hosted=True)
    assert app.HOSTED is True
    print(f"PASS hosted flag on (find-stock deps: {app._FIND_STOCK_OK}, "
          f"pipeline deps: {app._STOCK_PIPELINE_OK})")

    print("ALL HOSTED-MODE CHECKS PASSED")


if __name__ == "__main__":
    main()
