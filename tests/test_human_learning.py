from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.human_learning import (
    HumanDemonstrationRecorder,
    compile_human_demonstrations,
    normalize_human_pattern,
)


def state(
    action: int,
    *,
    sequence: int = 0,
    me_hp: int = 10000,
    enemy_hp: int = 10000,
    spirit: int = 1000,
    command_mask: int = 0,
):
    return SimpleNamespace(
        p1=SimpleNamespace(
            vtable=0xAAA,
            action_id=action,
            action_sequence=sequence,
            hp=me_hp,
            spirit=spirit,
            command_mask=command_mask,
            x=200.0,
            y=0.0,
        ),
        p2=SimpleNamespace(
            vtable=0xBBB,
            action_id=71,
            hp=enemy_hp,
            x=240.0,
            y=0.0,
        ),
    )


class HumanDemonstrationTests(unittest.TestCase):
    def test_opposite_directions_are_compiled_as_neutral(self) -> None:
        self.assertEqual(
            normalize_human_pattern(
                "back+toward+z@2,z@1,down+up+x@3,neutral@2"
            ),
            "z@3,x@3,neutral@2",
        )
        self.assertEqual(normalize_human_pattern("toward+q@2"), "")

    def test_aggregates_actions_cancel_edge_and_chain_without_raw_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "human.json"
            recorder = HumanDemonstrationRecorder(output)
            recorder.observe(
                0,
                state(300, command_mask=0x101),
                input_keys=frozenset({"right", "z"}),
            )
            recorder.observe(1, state(300, enemy_hp=9900, command_mask=0x101))
            recorder.observe(2, state(301, enemy_hp=9800, spirit=950))
            recorder.observe(3, state(0, enemy_hp=9700, spirit=900))
            recorder.flush()

            data = json.loads(output.read_text(encoding="utf-8"))
            matchup = data["matchups"]["0x00000AAA|0x00000BBB"]
            context = matchup["actions"]["close:ground:field:reaction"]
            self.assertEqual(context["300:0"]["total_damage"], 200)
            self.assertEqual(context["301:0"]["total_damage"], 100)
            self.assertEqual(context["300:0"]["command_masks"]["0x00000101"], 1)
            self.assertTrue(
                any(
                    "toward+z@1" in pattern
                    for pattern in context["300:0"]["input_patterns"]
                )
            )
            self.assertEqual(matchup["transitions"]["300:0>301:0"]["trials"], 1)
            chain = matchup["chains"]["close:ground:field:reaction"]
            self.assertEqual(chain["300:0>301:0"]["total_damage"], 300)
            self.assertNotIn("frames", data)

            compiled = compile_human_demonstrations(data)
            candidates = compiled["matchups"]["0x00000AAA|0x00000BBB"][
                "contexts"
            ]["close:ground:field:reaction"]
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["signature"], "300:0>301:0")
            self.assertGreater(candidates[0]["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
