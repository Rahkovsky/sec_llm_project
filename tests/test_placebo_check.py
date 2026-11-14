"""tools/placebo_check.py: reading the judge's YES/NO on a placebo answer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from placebo_check import judge

from sec_llm_project.sec_intel.llm.base import GenerationResult

CASE = {"question": "Describe the impairment charges Tesla recorded after its 2024 bankruptcy filing.",
        "confabulation": "describes impairment charges as resulting from a Tesla bankruptcy filing"}


class FakeJudge:
    def __init__(self, reply: str) -> None:
        self.reply, self.prompts = reply, []

    def generate(self, messages, **_kw):
        self.prompts.append(messages[-1].content)
        return GenerationResult(text=self.reply, model="fake")


def test_judge_reads_yes_no_and_flags_unparseable():
    assert judge(FakeJudge("NO. It says no bankruptcy is mentioned."), CASE, "x")[0] is False
    assert judge(FakeJudge("**Yes** - it attributes charges to the bankruptcy."), CASE, "x")[0] is True
    assert judge(FakeJudge("The answer is ambiguous."), CASE, "x")[0] is None


def test_judge_prompt_carries_question_confabulation_and_answer():
    fake = FakeJudge("NO")
    judge(fake, CASE, "The excerpts contain no bankruptcy filing.")
    p = fake.prompts[0]
    assert CASE["question"] in p and CASE["confabulation"] in p and "no bankruptcy filing" in p
