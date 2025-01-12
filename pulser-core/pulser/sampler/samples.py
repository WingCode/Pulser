# Copyright 2022 Pulser Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines samples dataclasses."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pulser.sequence import QubitId


@dataclass
class QubitSamples:
    """Gathers samples concerning a single qubit."""

    amp: np.ndarray
    det: np.ndarray
    phase: np.ndarray
    qubit: QubitId

    def __post_init__(self) -> None:
        if not len(self.amp) == len(self.det) == len(self.phase):
            raise ValueError(
                "ndarrays amp, det and phase must have the same length."
            )
