# SPDX-License-Identifier: Apache-2.0
"""The queue gate must fail closed on every incomplete matrix result."""

import os
import re
import subprocess
import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/test.yml"


class MergeQueueGateTests(unittest.TestCase):
    def test_queue_trigger_and_gate_contract(self):
        text = WORKFLOW.read_text()
        self.assertIn("  merge_group:\n    types: [checks_requested]", text)
        gate = text.split("  verification-gate:\n", 1)[1]
        self.assertIn("    name: Robot verification gate\n", gate)
        self.assertIn("    if: always()\n", gate)
        self.assertIn("    needs: [test]\n", gate)
        self.assertIn("TEST_RESULT: ${{ needs.test.result }}", gate)
        self.assertNotIn("continue-on-error", text)
        self.assertNotIn("paths:", text)
        self.assertNotIn("paths-ignore:", text)

    def test_actual_gate_command_rejects_incomplete_results(self):
        gate = WORKFLOW.read_text().split("  verification-gate:\n", 1)[1]
        command = re.search(r"^        run: (.+)$", gate, re.M).group(1)
        for result in ("success", "failure", "cancelled", "skipped", "neutral", "", "unknown"):
            with self.subTest(result=result):
                process = subprocess.run(
                    ["bash", "-eu", "-c", command],
                    env={**os.environ, "TEST_RESULT": result},
                    check=False,
                )
                self.assertEqual(process.returncode == 0, result == "success")


if __name__ == "__main__":
    unittest.main()
