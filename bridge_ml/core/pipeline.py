from typing import Dict, List, Optional

class MatchingPipeline:
    def __init__(self, steps: List[str]):
        self.steps = steps

    def run(self, data: Dict) -> Dict:
        """
        Execute the matching pipeline.
        """
        raise NotImplementedError
