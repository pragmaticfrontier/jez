"""Benchmark tasks mapped onto Jev questions.

Every task turns one dataset row into a Jev request (``state`` plus one question) and a
gold answer key: an option name for ``choice``, ``"true"``/``"false"`` for ``noul``, a level
index as a string for ``score``.

Tier 1 covers the thirteen datasets with a published Jev figure. Tier 2 (typed-decisions,
Decision Bench, the Laya battery, withdrawal routing) is read from local JSONL files whose
rows already hold ``state``, ``question`` and ``gold``; see ``eval/data/tier2/README.md``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .data import label_names, load_rows

Example = tuple[Any, dict[str, Any], str]  # (state, question, gold)


@dataclass
class Task:
    name: str
    dataset: str
    config: str
    split: str
    qtype: str  # noul | choice | score
    build: Callable[[dict], Example]
    metric: str = "accuracy"  # accuracy | f1_positive
    tier: int = 1
    language: str | None = None
    group: str | None = None  # tasks that are averaged together (Belebele languages)
    description: str = ""
    local: Path | None = None  # Tier 2: a JSONL file instead of a Hub dataset
    extra: dict = field(default_factory=dict)

    def rows(self, n: int | None) -> list[dict]:
        if self.local is not None:
            if not self.local.exists():
                raise FileNotFoundError(f"{self.name}: put the rows in {self.local} (see eval/data/tier2/README.md)")
            rows = [json.loads(line) for line in self.local.read_text().split("\n") if line.strip()]
            return rows if n is None else rows[:n]
        return load_rows(self.name, self.dataset, self.config, self.split, n)


# ----------------------------------------------------------------------------- helpers

def _choice(instructions: str, criteria: dict[str, Any]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _noul(instructions: str, true: str | None = None, false: str | None = None) -> dict[str, Any]:
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true or false:
        q["criteria"] = {"true": true, "false": false}
    return q


def _score(instructions: str, levels: list[str]) -> dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        n = n.strip() or "(empty)"
        if n in seen:
            seen[n] += 1
            n = f"{n} ({seen[n]})"
        else:
            seen[n] = 1
        out.append(n)
    return out


# ----------------------------------------------------------------------------- Tier 1 builders

SENTIMENT = {"negative": "The text expresses a negative opinion", "positive": "The text expresses a positive opinion"}


def sst2(r: dict) -> Example:
    return r["sentence"].strip(), _choice("What is the sentiment of this movie review sentence?", SENTIMENT), \
        ["negative", "positive"][int(r["label"])]


def imdb(r: dict) -> Example:
    return r["text"].strip(), _choice("What is the sentiment of this movie review?", SENTIMENT), \
        ["negative", "positive"][int(r["label"])]


def hellaswag(r: dict) -> Example:
    endings = _dedupe(list(r["endings"]))
    state = {"activity": r.get("activity_label", ""), "context": r["ctx"].strip()}
    return state, _choice("Which ending continues the context most plausibly?", {e: None for e in endings}), \
        endings[int(r["label"])]


def arc(r: dict) -> Example:
    texts = _dedupe(list(r["choices"]["text"]))
    labels = list(r["choices"]["label"])
    gold = texts[labels.index(r["answerKey"])]
    return r["question"].strip(), _choice("Which answer to the question is correct?", {t: None for t in texts}), gold


def boolq(r: dict) -> Example:
    q = r["question"].strip().rstrip("?")
    q = q[:1].upper() + q[1:] + "?"
    return r["passage"].strip(), _noul(q), "true" if r["answer"] in (True, "True", "true", 1) else "false"


def belebele(r: dict) -> Example:
    answers = _dedupe([r["mc_answer1"], r["mc_answer2"], r["mc_answer3"], r["mc_answer4"]])
    state = {"passage": r["flores_passage"].strip(), "question": r["question"].strip()}
    return state, _choice("Which answer to the question is correct according to the passage?", {a: None for a in answers}), \
        answers[int(r["correct_answer_num"]) - 1]


AGNEWS = {"world": "World news: international affairs, politics, conflicts",
          "sports": "Sports: games, athletes, teams, results",
          "business": "Business: companies, markets, finance, economy",
          "sci_tech": "Science and technology: research, software, hardware, space"}


def agnews(r: dict) -> Example:
    return r["text"].strip(), _choice("Which section of a newspaper does this article belong to?", AGNEWS), \
        list(AGNEWS)[int(r["label"])]


EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def emotion(r: dict) -> Example:
    return r["text"].strip(), _choice("Which emotion does the writer of this text express?", {e: None for e in EMOTIONS}), \
        EMOTIONS[int(r["label"])]


def banking77(r: dict) -> Example:
    names = label_names("legacy-datasets/banking77", "default")  # PolyAI/banking77 is a script dataset the rows API cannot serve
    return r["text"].strip(), _choice("Which banking intent does this customer message express?",
                                      {n: n.replace("_", " ") for n in names}), names[int(r["label"])]


def toxicchat(r: dict) -> Example:
    return r["user_input"].strip(), _noul("Is this user message toxic?",
                                          "Harassment, hate, sexual content, violence, self-harm or other toxic content",
                                          "An ordinary, non-toxic request or conversation"), \
        "true" if int(r["toxicity"]) == 1 else "false"


def prompt_injection(r: dict) -> Example:
    return r["text"].strip(), _noul("Is this text a prompt injection attempt?",
                                    "It tries to override, ignore or subvert the system's instructions",
                                    "It is a legitimate request or statement"), \
        "true" if int(r["label"]) == 1 else "false"


SST5_LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]


def sst5(r: dict) -> Example:
    return r["text"].strip(), _score("How positive is the sentiment of this movie review sentence?", SST5_LEVELS), \
        str(int(r["label"]))


# ----------------------------------------------------------------------------- registry

BELEBELE_LANGUAGES = ["eng_Latn", "deu_Latn", "fra_Latn", "spa_Latn", "ita_Latn", "por_Latn",
                      "rus_Cyrl", "ukr_Cyrl", "arb_Arab", "hin_Deva", "zho_Hans", "jpn_Jpan"]

TIER2_DIR = Path(__file__).parent / "data" / "tier2"


def tier2(name: str, qtype: str, description: str, metric: str = "accuracy") -> Task:
    """A local JSONL task whose rows hold state, question and gold already in Jev format."""
    return Task(name=name, dataset="local", config="", split="", qtype=qtype, tier=2, description=description,
                metric=metric, local=TIER2_DIR / f"{name}.jsonl",
                build=lambda r: (r["state"], r["question"], str(r["gold"])))


TASKS: dict[str, Task] = {
    "sst2": Task("sst2", "stanfordnlp/sst2", "default", "validation", "choice", sst2, description="SST-2 sentiment"),
    "imdb": Task("imdb", "stanfordnlp/imdb", "plain_text", "test", "choice", imdb, description="IMDB sentiment"),
    "hellaswag": Task("hellaswag", "Rowan/hellaswag", "default", "validation", "choice", hellaswag,
                      description="HellaSwag commonsense endings"),
    "arc_easy": Task("arc_easy", "allenai/ai2_arc", "ARC-Easy", "test", "choice", arc, description="ARC-Easy science questions"),
    "arc_challenge": Task("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "test", "choice", arc,
                          description="ARC-Challenge science questions"),
    "boolq": Task("boolq", "google/boolq", "default", "validation", "noul", boolq, description="BoolQ reading comprehension"),
    "agnews": Task("agnews", "fancyzhx/ag_news", "default", "test", "choice", agnews, description="AG News topic"),
    "emotion": Task("emotion", "dair-ai/emotion", "split", "test", "choice", emotion, description="DAIR Emotion"),
    "banking77": Task("banking77", "legacy-datasets/banking77", "default", "test", "choice", banking77, description="BANKING77 intents"),
    "toxicchat": Task("toxicchat", "lmsys/toxic-chat", "toxicchat0124", "test", "noul", toxicchat, metric="f1_positive",
                      description="ToxicChat toxicity (F1 on toxic)"),
    "prompt_injection": Task("prompt_injection", "deepset/prompt-injections", "default", "test", "noul", prompt_injection,
                             description="deepset prompt injections"),
    "sst5": Task("sst5", "SetFit/sst5", "default", "test", "score", sst5, description="SST-5 sentiment (5 levels)"),
}
for lang in BELEBELE_LANGUAGES:
    TASKS[f"belebele_{lang}"] = Task(f"belebele_{lang}", "facebook/belebele", lang, "test", "choice", belebele,
                                     language=lang, group="belebele", description=f"Belebele {lang}")

TASKS["typed_decisions"] = tier2("typed_decisions", "mixed", "typed-decisions test split (in-domain for Jez)")
TASKS["decision_bench"] = tier2("decision_bench", "mixed", "Decision Bench bench-v4, text rows")
TASKS["laya_battery"] = tier2("laya_battery", "mixed", "Laya battery, 5 held-out tasks")
TASKS["withdrawal_routing"] = tier2("withdrawal_routing", "choice", "withdrawal routing, 500 cases, order stability")

TIER1 = [t for t in TASKS.values() if t.tier == 1]
TIER1_NAMES = ["sst2", "imdb", "hellaswag", "arc_easy", "arc_challenge", "boolq", "belebele", "agnews", "emotion",
               "banking77", "toxicchat", "prompt_injection", "sst5"]


def expand(names: list[str]) -> list[Task]:
    """Resolve task names, expanding ``belebele`` and ``tier1`` / ``tier2``."""
    out: list[Task] = []
    for n in names:
        if n == "tier1":
            out.extend(TIER1)
        elif n == "tier2":
            out.extend(t for t in TASKS.values() if t.tier == 2)
        elif n == "belebele":
            out.extend(t for t in TASKS.values() if t.group == "belebele")
        elif n in TASKS:
            out.append(TASKS[n])
        else:
            raise KeyError(f"unknown task {n!r}; known: {sorted(TASKS)} plus tier1, tier2, belebele")
    return out
