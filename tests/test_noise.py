import math

import numpy as np
import pytest

from mujoco_module.sensors import Noise


def test_noise_is_seeded_and_biased():
    truth = np.array([1.0, 2.0, 3.0])
    a, b, c = Noise(7), Noise(7), Noise(8)
    first = [a.vector(truth, 0.5) for _ in range(3)]
    assert all(np.array_equal(x, b.vector(truth, 0.5)) for x in first)
    assert not np.array_equal(first[0], c.vector(truth, 0.5))
    samples = np.array([Noise(1).vector(truth, 0.0, [0.1, 0, 0])] + [a.vector(truth, 0.5, [0.1, 0, 0]) for _ in range(2000)])
    assert samples[0] == pytest.approx([1.1, 2.0, 3.0])  # no noise: exactly truth plus bias
    assert samples[1:].mean(axis=0) == pytest.approx([1.1, 2.0, 3.0], abs=0.05)
    assert samples[1:].std(axis=0) == pytest.approx([0.5, 0.5, 0.5], rel=0.08)
    angles = [
        2 * math.degrees(math.acos(min(1.0, abs(Noise(s).rotation(np.array([1.0, 0, 0, 0]), 2.0)[0]))))
        for s in range(500)
    ]
    assert np.sqrt(np.mean(np.square(angles))) == pytest.approx(2.0, rel=0.1)
    assert np.array_equal(Noise(1).rotation(np.array([1.0, 0, 0, 0]), 0.0), [1.0, 0, 0, 0])
