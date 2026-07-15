import unittest

import torch

from utils.proto_model import _compute_geometry_rho, _prototype_geometry_pressure


class GeometryRhoTests(unittest.TestCase):
    def test_one_shot_crowding_is_nonzero_and_ordered(self):
        labels = torch.tensor([0, 1])
        separated = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        crowded = torch.tensor([[1.0, 0.0], [0.9, 0.1]])

        separated_pressure = _prototype_geometry_pressure(separated, labels)
        crowded_pressure = _prototype_geometry_pressure(crowded, labels)

        self.assertGreater(float(separated_pressure["pressure"]), 0.0)
        self.assertGreater(
            float(crowded_pressure["pressure"]),
            float(separated_pressure["pressure"]),
        )

    def test_protective_mapping_boosts_mid_and_shrinks_high_pressure(self):
        mid_rho = _compute_geometry_rho(
            0.1, 0.2, 0.15, 0.75, 1.15, 0.35, 0.75
        )
        high_rho = _compute_geometry_rho(
            0.1, 0.8, 0.15, 0.75, 1.15, 0.35, 0.75
        )
        self.assertGreater(mid_rho, 0.1)
        self.assertLess(high_rho, 0.1)
        self.assertGreaterEqual(high_rho, 0.075)


if __name__ == "__main__":
    unittest.main()
