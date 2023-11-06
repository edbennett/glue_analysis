from glue_analysis.correlator import CorrelatorEnsemble


def test_correlator_ensemble_stores_filename() -> None:
    assert CorrelatorEnsemble("filename").filename == "filename"
