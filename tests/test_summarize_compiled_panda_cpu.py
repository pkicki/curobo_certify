import unittest

from scripts.summarize_compiled_panda_cpu import quantile


class QuantileTests(unittest.TestCase):
    def test_linear_interpolated_quantile(self):
        self.assertEqual(quantile([1, 2, 3, 4], .5), 2.5)
        self.assertAlmostEqual(quantile([1, 2, 3, 4], .95), 3.85)


if __name__ == "__main__":
    unittest.main()
