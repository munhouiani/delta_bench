"""Hypothesis: 100 examples per property, or 1,000 with HYPOTHESIS_PROFILE=evidence."""

import os

from hypothesis import settings

settings.register_profile("default", max_examples=100, deadline=None)
settings.register_profile("evidence", max_examples=1000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
