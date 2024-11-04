import pytest
from bridge_ml.core.pipeline import MatchingPipeline

def test_pipeline_initialization():
    pipeline = MatchingPipeline(steps=['preprocess', 'match'])
    assert pipeline.steps == ['preprocess', 'match']
