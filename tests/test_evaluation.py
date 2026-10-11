import pytest
from src.services.evaluation_service import evaluation_runner
from src.utils.llm_client import llm_client


def test_evaluation_benchmark():
    """
    Tests run_benchmark() in whichever mode is currently active:
    - offline_no_llm (no OPENAI_API_KEY): LLM metrics are SKIPPED_NO_LLM_KEY
      with null scores; benchmark_status is NOT_EVALUATED.
    - live_llm_judge (API key set): real DeepEval scores are present;
      benchmark_status is PASSED or NEEDS_OPTIMIZATION.
    """
    summary = evaluation_runner.run_benchmark()

    # Fields always present regardless of mode
    assert summary["total_scenarios_evaluated"] >= 5
    assert "evaluation_mode" in summary
    assert summary["evaluation_mode"] in ("offline_no_llm", "live_llm_judge")
    assert "mean_scores" in summary
    assert "benchmark_status" in summary
    assert "evaluation_framework" in summary

    # Version string must be dynamic (not hardcoded)
    import deepeval as _dv
    assert _dv.__version__ in summary["evaluation_framework"]

    # Valuation accuracy is always computed (rule-based, no LLM required)
    mean_val_acc = summary["mean_scores"]["mean_valuation_accuracy"]
    assert mean_val_acc is not None
    assert 0.0 <= float(mean_val_acc) <= 1.0

    if not llm_client.is_live:
        # Offline mode assertions
        assert summary["evaluation_mode"] == "offline_no_llm"
        assert summary["benchmark_status"] == "NOT_EVALUATED"
        assert summary["pass_rate_percentage"] is None
        assert summary["passed_scenarios"] is None

        # All LLM-backed metrics must be null
        for key in ("mean_faithfulness", "mean_answer_relevancy",
                    "mean_contextual_precision", "mean_contextual_recall"):
            assert summary["mean_scores"][key] is None, (
                f"Expected null for {key} in offline mode but got "
                f"{summary['mean_scores'][key]}"
            )

        # All scenario results must have SKIPPED status for LLM metrics
        for result in summary["scenario_results"]:
            if "error" not in result:
                assert result.get("deepeval_metric_status") == "SKIPPED_NO_LLM_KEY"
                for metric_key in ("faithfulness", "answer_relevancy",
                                   "contextual_precision", "contextual_recall"):
                    assert result["metrics"][metric_key] is None, (
                        f"Expected null {metric_key} in offline scenario result"
                    )
    else:
        # Live mode assertions
        assert summary["evaluation_mode"] == "live_llm_judge"
        assert summary["benchmark_status"] in ("PASSED", "NEEDS_OPTIMIZATION")
        assert summary["pass_rate_percentage"] is not None
        assert summary["passed_scenarios"] is not None

        # LLM-backed mean scores should be real floats
        for key in ("mean_faithfulness", "mean_answer_relevancy",
                    "mean_contextual_precision", "mean_contextual_recall"):
            val = summary["mean_scores"][key]
            assert val is not None, f"Expected non-null {key} in live mode"
            assert 0.0 <= float(val) <= 1.0
