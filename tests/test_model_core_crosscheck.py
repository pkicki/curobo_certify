"""Cross-language exactness: browser model_core.js vs Python exact Fraction reference."""
import base64, json, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify.exact_k3_reference import bernstein_coeffs, build_residuals  # noqa: E402
from curobo_certify.native_error_bound import exact_timing_table  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402


def node_cmd():
    for cand in (["node"], ["conda", "run", "-n", "timeopt-report-tests", "node"]):
        if shutil.which(cand[0]):
            try:
                subprocess.run(cand + ["--version"], check=True, capture_output=True)
                return cand
            except Exception:
                pass
    return None


def table_strings():
    t = exact_timing_table()  # [m][P=3-p][k]
    return [[[str(t[m][3 - p][k]) for k in range(4)] for p in range(4)] for m in range(29)]


class ModelCoreCrossCheck(unittest.TestCase):
    def test_js_residuals_equal_python_exact(self):
        cmd = node_cmd()
        if cmd is None:
            self.skipTest("node unavailable")
        rng = np.random.default_rng(11)
        D = 3
        start, goal = rng.normal(size=D), rng.normal(size=D)
        inner = [start + (goal - start) * (k + 1) / 7 + 0.05 * rng.normal(size=D) for k in range(6)]
        geo = np.array([start] * 3 + inner + [goal] * 3, dtype=np.float32)
        timing = (3 + 20 * rng.random(32)).astype(np.float32)
        lim = {"velocity": [2.1, 1.7, 3.0], "acceleration": [15.0, 9.5, 20.0], "jerk": [7500.0, 3750.0, 5000.0]}
        spans = extract_timeopt_spans(geo, timing)
        queries = [(si, fam, j) for si in (0, 3, len(spans) // 2, len(spans) - 1)
                   for fam, j in (("positive_x", -1), ("velocity", 1), ("acceleration_plus", 0),
                                  ("acceleration_minus", 2), ("jerk", 1))]
        from fractions import Fraction
        exact_lim = {k: [str(Fraction(float(v))) for v in vs] for k, vs in lim.items()}
        case = {"path": {"shape": list(geo.shape), "geometry_f32": base64.b64encode(geo.astype("<f4").tobytes()).decode(),
                         "timing_f32": base64.b64encode(timing.astype("<f4").tobytes()).decode(), "limits": exact_lim},
                "table": table_strings(), "queries": queries}
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "case.json"
            f.write_text(json.dumps(case))
            res = json.loads(subprocess.run(cmd + [str(ROOT / "tests/model_core_crosscheck.cjs"), str(f)],
                                            check=True, capture_output=True, text=True).stdout)
        self.assertEqual(res["spans"], len(spans))
        for (si, fam, j), js in zip(queries, res["coefficients"]):
            py = None
            for name, joint, power in build_residuals(spans[si], lim["velocity"], lim["acceleration"], lim["jerk"]):
                if name == fam and joint == j:
                    py = [str(c) for c in bernstein_coeffs(power)]
            self.assertEqual(js, py, (si, fam, j))


if __name__ == "__main__":
    unittest.main()
