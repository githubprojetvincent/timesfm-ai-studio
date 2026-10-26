import importlib.util
from pathlib import Path
import unittest
import numpy as np

SPEC = importlib.util.spec_from_file_location("timesfm_app", Path(__file__).resolve().parents[1] / "core.py")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)

class ForecastCoreTests(unittest.TestCase):
    def test_forecast_fallback_returns_horizon(self):
        x = np.arange(40, dtype=float) * 0.5 + 10
        pred, bounds, engine = core.forecast_series(x, 12, model_loader=lambda: (_ for _ in ()).throw(RuntimeError("offline")))
        self.assertEqual(len(pred), 12)
        self.assertEqual(bounds.shape, (12, 2))
        self.assertIn("secours", engine.lower())
        self.assertTrue(np.isfinite(pred).all())
    def test_rejects_short_series(self):
        with self.assertRaises(ValueError):
            core.forecast_series(np.arange(5, dtype=float), 3, model_loader=lambda: None)
    def test_horizon_is_clamped(self):
        self.assertEqual(core.validate_horizon(9999), 512)
        self.assertEqual(core.validate_horizon(0), 1)

    def test_ml_requires_enough_observations(self):
        with self.assertRaises(ValueError):
            core.machine_learning_forecast(np.arange(20, dtype=float), 5)

    def test_ml_forecast_returns_finite_values_and_scores(self):
        x = np.arange(70, dtype=float)
        y = 10 + 0.2*x + 2*np.sin(x/4)
        result = core.machine_learning_forecast(y, 5)
        self.assertEqual(len(result["forecast"]), 5)
        self.assertTrue(np.isfinite(result["forecast"]).all())
        self.assertGreaterEqual(len(result["scores"]), 2)
        self.assertIn("mae", result)

if __name__ == "__main__":
    unittest.main()
