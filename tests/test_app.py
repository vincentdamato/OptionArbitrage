from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.mark.parametrize("choice", [0, 1])
def test_app_renders_each_demo_without_errors(choice):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(at.sidebar.radio[0].options[choice]).run()
    assert not at.exception
    assert any("Synthetic data" in i.value for i in at.info)


def test_demo_shows_planted_violations():
    at = AppTest.from_file(APP, default_timeout=60).run()
    labels = {m.label: m.value for m in at.metric}
    assert int(labels["Model-free violations"]) > 0


def test_app_has_no_credential_inputs_or_order_buttons():
    source = Path(APP).read_text().lower()
    for word in ("password", "api_key", "secret", "execute", "place order"):
        assert word not in source


def test_schwab_format_demo_shows_vendor_greek_comparison():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(at.sidebar.radio[0].options[2]).run()
    assert not at.exception
    assert any("Model-built sample" in i.value for i in at.info)
    assert any("Against the source's own IV and Greeks" in h.value for h in at.subheader)
    labels = {m.label: m.value for m in at.metric}
    assert labels["Rate used"] == "4.25%"
