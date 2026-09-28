"""The one replaceable part of the pipeline.

Everything downstream talks to this protocol and nothing else, so swapping
Supertonic for another engine is a new file implementing `SynthesisEngine`
plus a one-line change at the call site. `engine_id` takes part in the clip
cache key, so two engines' outputs can coexist in the cache without colliding.
"""

from typing import Protocol

import numpy as np


class SynthesisEngine(Protocol):
    engine_id: str
    sample_rate: int

    def synthesize(self, text: str, language: str, voice: str) -> np.ndarray:
        """Return mono float32 samples in [-1.0, 1.0] at `self.sample_rate`."""
        ...
