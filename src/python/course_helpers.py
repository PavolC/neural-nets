"""Helpers the course provides to exercises via `from course import ...`.

These are functions the learner has already built in an earlier module (or
that the course supplies as given). The harness registers this file as the
`course` module inside Pyodide, so skeletons never contain solution logic
for the current exercise (see CLAUDE.md hard rules).
"""

import numpy as np


def sigmoid(z):
    """The sigmoid function 1 / (1 + exp(-z)), applied elementwise."""
    return 1.0 / (1.0 + np.exp(-z))
