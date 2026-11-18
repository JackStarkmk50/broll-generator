"""Hosted-mode UI behaviour via Streamlit AppTest.

Run:  python tests/test_hosted_ui.py   (slim install is enough)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile
TMP = tempfile.mkdtemp(prefix="edit-helper-test-")


def use_temp_db():
    """Throwaway DB so the app under test never touches a real checkout's data.

    db.database reads the module-global DB_PATH when the Database singleton is
    created, so patch it there and reset the singleton - patching
    config.settings alone would not move the database.
    """
    import db.database as dbm
    dbm.DB_PATH = Path(TMP) / "data" / "edit_helper.db"
    dbm.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    dbm._db_instance = None
    import config.settings as cs
    cs.settings.TEMP_DIR = Path(TMP) / "temp_cache"
    cs.settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)


def run_app(hosted, code=None):
    os.environ.pop("EDIT_HELPER_HOSTED", None)
    os.environ.pop("EDIT_HELPER_ACCESS_CODE", None)
    if hosted:
        os.environ["EDIT_HELPER_HOSTED"] = "1"
    if code:
        os.environ["EDIT_HELPER_ACCESS_CODE"] = code
    use_temp_db()
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "gui" / "app.py"), default_timeout=30)
    at.run()
    enter_app(at)
    return at


def enter_app(at):
    """Click through the landing page and create a project to reach the nav."""
    for b in at.button:
        if b.key == "landing_try":
            b.click().run()
            break
    # sidebar form: name the project and submit
    for ti in at.text_input:
        if ti.label == "Project Name":
            ti.set_value("Test Project")
    for b in at.button:
        if b.label == "Create":
            b.click().run()
            break
    at.run()


def main():
    from streamlit import config as st_config

    # local mode: no hosted warning, config.toml limit untouched
    at = run_app(hosted=False)
    warns = [w.value for w in at.warning]
    assert not any("anyone with the link" in w for w in warns), warns
    assert st_config.get_option("server.maxUploadSize") == 10000
    print("PASS local: no hosted warning, upload limit stays at config.toml value")

    # hosted with access code: gate works
    os.environ["EDIT_HELPER_HOSTED"] = "1"
    os.environ["EDIT_HELPER_ACCESS_CODE"] = "test123"
    use_temp_db()
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "gui" / "app.py"), default_timeout=30)
    at.run()
    assert len(at.text_input) == 1
    at.text_input[0].set_value("wrong").run()
    assert any("Wrong access code" in e.value for e in at.error)
    at.text_input[0].set_value("test123").run()
    at.run()
    assert len(at.error) == 0 and len(at.markdown) > 0
    print("PASS hosted with code: gate blocks wrong code, admits right code")

    # hosted, no access code: fail closed, nothing else renders
    at = run_app(hosted=True)
    errs = [e.value for e in at.error]
    assert any("not open yet" in e for e in errs), f"fail-closed error missing: {errs}"
    assert len(at.button) == 0 and len(at.selectbox) == 0, "app must not render past the closed gate"
    assert st_config.get_option("server.maxUploadSize") == 200, "hosted must cap uploads at 200"
    print("PASS hosted, no code: fails closed with explanation, nothing rendered")

    # hosted with code: legacy pages hidden after entering
    at = run_app(hosted=True, code="test123")
    at.text_input[0].set_value("test123").run()
    enter_app(at)
    caps = [c.value for c in at.caption]
    assert any("Find stock and Stock pipeline are unavailable" in c for c in caps), caps
    opts = [o for sb in at.selectbox for o in (sb.options or [])]
    assert "Find stock" not in opts and "Stock pipeline" not in opts, opts
    assert "Create images" in opts and "Library" in opts, opts
    caps = [c.value for c in at.caption]
    assert any("keys are never saved" in c for c in caps), caps
    print("PASS hosted with code: legacy pages hidden, honest key caption shown")

    # seeded DB: hosted sidebar must NOT prefill saved keys into the inputs
    os.environ["EDIT_HELPER_HOSTED"] = "1"
    os.environ["EDIT_HELPER_ACCESS_CODE"] = "test123"
    use_temp_db()
    from db.database import get_db
    seeded_db = get_db()
    seeded_db.set_setting("gemini_api_key", "SAVED-DB-KEY")
    seeded_db.set_setting("pexels_api_key", "SAVED-DB-PEXELS")
    seeded_db.set_setting("llm_backend", "api")  # Gemini input only renders for the api backend
    from streamlit.testing.v1 import AppTest as _AT
    at = _AT.from_file(str(Path(__file__).resolve().parent.parent / "gui" / "app.py"), default_timeout=30)
    at.run()
    at.text_input[0].set_value("test123").run()
    enter_app(at)
    key_inputs = {ti.label: ti.value for ti in at.text_input if "API Key" in (ti.label or "")}
    assert "Pexels API Key" in key_inputs and "Gemini API Key" in key_inputs, key_inputs
    assert key_inputs["Pexels API Key"] == "", key_inputs
    assert key_inputs["Gemini API Key"] == "", key_inputs
    print("PASS hosted with seeded DB: key inputs render empty, no DB prefill")

    print("ALL HOSTED UI CHECKS PASSED")


if __name__ == "__main__":
    main()
