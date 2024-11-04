import pytest
from bridge_ml.core.matcher import OntologyMatcher

def test_matcher_initialization():
    matcher = OntologyMatcher()
    assert matcher is not None
