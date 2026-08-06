import pandas as pd
import pytest

from portpy.explain import Explanation, MetricResult, available, explain, get, register


def test_metric_result_behaves_like_float():
    result = MetricResult(1.5, "sharpe_ratio")
    assert result == 1.5
    assert result + 1 == 2.5
    assert result * 2 == 3.0
    assert isinstance(result, float)
    assert float(result) == 1.5


def test_metric_result_repr_includes_name_and_interpretation():
    result = MetricResult(1.5, "sharpe_ratio")
    text = repr(result)
    assert "sharpe_ratio" in text
    assert "good" in text  # 1.5 falls in the "good" bucket per registered interpret()


def test_metric_result_str_is_plain_number():
    result = MetricResult(1.5, "sharpe_ratio")
    assert str(result) == str(1.5)


def test_metric_result_explain_returns_full_card(capsys):
    result = MetricResult(1.5, "sharpe_ratio")
    text = result.explain(print_it=False)
    assert "sharpe_ratio" in text
    assert "This result:" in text
    captured = capsys.readouterr()
    assert captured.out == ""  # print_it=False should not print


def test_register_and_get_roundtrip():
    expl = Explanation(
        name="_test_metric_xyz",
        category="metric",
        summary="A test metric.",
        how_to_read="Read it as a test.",
        good_vs_bad="Higher is better, for testing.",
    )
    register(expl)
    assert get("_test_metric_xyz") is expl
    assert "_test_metric_xyz" in available("metric")


def test_get_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        get("_definitely_not_registered_xyz")


def test_explain_dispatches_on_string_name():
    text = explain("sharpe_ratio", print_it=False)
    assert "sharpe_ratio" in text


def test_explain_dispatches_on_metric_result():
    result = MetricResult(2.5, "sharpe_ratio")
    text = explain(result, print_it=False)
    assert "very good" in text


def test_explain_dispatches_on_pandas_attrs():
    df = pd.DataFrame({"a": [1, 2]})
    df.attrs["portpy_explanation"] = "sharpe_ratio"
    text = explain(df, print_it=False)
    assert "sharpe_ratio" in text


def test_explain_unknown_object_raises_type_error():
    with pytest.raises(TypeError):
        explain(12345, print_it=False)


def test_explanation_invalid_category_raises():
    with pytest.raises(ValueError):
        Explanation(
            name="_bad",
            category="not_a_category",
            summary="x",
            how_to_read="x",
            good_vs_bad="x",
        )


def test_package_level_explain_api_is_callable_and_exposes_registry_helpers():
    import portpy as pt

    assert callable(pt.explain)
    assert "sharpe_ratio" in pt.explain.available()
    assert pt.explain.get("sharpe_ratio").name == "sharpe_ratio"
