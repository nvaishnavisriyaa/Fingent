"""Keep the acceptance tests deterministic and offline.

The native tools call live APIs by default (FINGENT_LIVE_DATA=1). For the test suite we force
the offline deterministic fallback so tests are fast and reproducible without network access.
"""
import os

os.environ["FINGENT_LIVE_DATA"] = "0"
