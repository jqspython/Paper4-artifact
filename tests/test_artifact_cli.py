import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import paper4_artifact_cli as cli


class ArtifactCliTests(unittest.TestCase):
    def capture(self, function, arguments):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            function(arguments)
        return json.loads(stream.getvalue())

    def test_synthetic_command(self):
        result = self.capture(cli.synthetic_main, [])
        self.assertEqual(result["status"], "PASS_SYNTHETIC_ONLY")
        self.assertAlmostEqual(result["identity_residual"], 0.)

    def test_replay_base_and_wrapper(self):
        wrapped = cli.replay_records(cli._records(), capacity_rps=10., wrapped=True)
        base = cli.replay_records(cli._records(), capacity_rps=10., wrapped=False)
        self.assertTrue(wrapped["snapshot"]["arithmetic_ok"])
        self.assertIsNone(base["snapshot"]["arithmetic_ok"])

    def test_csv_monitor_and_single_wrapper_clis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ticks.csv"
            path.write_text("y_count,n_start,base_request,runtime_cap\n20,1,2,8\n", encoding="utf-8")
            monitor = self.capture(cli.monitor_main, ["--input", str(path)])
            wrapper = self.capture(cli.wrapper_main, ["--y-count", "20", "--n-start", "1", "--base-request", "2"])
        self.assertTrue(monitor["contract_valid"])
        self.assertEqual(wrapper["emitted"], [2])

    def test_container_example_is_synthetic_only(self):
        root = Path(__file__).resolve().parents[1]
        dockerfile = (root / "Dockerfile.artifact").read_text(encoding="utf-8")
        manifest = (root / "sysbench/kind-artifact-check.yaml").read_text(encoding="utf-8")
        self.assertIn('ENTRYPOINT ["paper4-reproduce-synthetic"]', dockerfile)
        self.assertIn('command: ["paper4-reproduce-synthetic"]', manifest)
        self.assertNotIn("hostPath", manifest)
        self.assertNotIn("volumeMounts", manifest)


if __name__ == "__main__":
    unittest.main()
