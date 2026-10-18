"""Candidate training sources, their licences, and how each row becomes Jev questions.

Every builder turns one dataset row into a list of *sibling* questions about the same state
(for example a choice over intents plus a yes/no about one intent). ``train.build_mix`` keeps all
siblings as one multi-question request for 30% of rows and one sibling otherwise, so every
multi-question request is several questions about one state and never a collage of rows.

Nothing here overlaps the evaluation suite by construction (no SST, IMDB, HellaSwag, ARC, BoolQ,
Belebele, AG News, Emotion, BANKING77, ToxicChat, deepset injections or SST-5), and
``train.build_mix`` still drops any state sharing a 13-token window with an evaluation item.

Licences come from each dataset's Hub card (mirrors carry the original dataset's licence).
``commercial_ok`` is False for research-only and non-commercial licences and ``None`` where the
card does not state terms clearly; both stay out of the mix unless ``--allow-noncommercial`` is
passed (a non-commercial Jez release) or the terms have been checked and the flag flipped here.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

Example = dict[str, Any]  # {"state", "question", "gold", "source", "language"}
Builder = Callable[[dict, random.Random], list[Example]]


@dataclass
class Source:
    name: str
    dataset: str
    config: str
    split: str
    licence: str
    commercial_ok: bool | None
    build: Builder
    qtypes: tuple[str, ...]
    languages: tuple[str, ...] = ("en",)
    description: str = ""
    needs: dict = field(default_factory=dict)  # auxiliary lists to attach to every row (see build_mix.iter_source)


def _ex(state, question, gold, source: str, language: str = "en") -> Example:
    return {"state": state, "question": question, "gold": str(gold), "source": source, "language": language}


def _choice(state, instructions: str, criteria: dict, gold: str, source: str, language: str = "en") -> Example:
    return _ex(state, {"type": "choice", "instructions": instructions, "criteria": criteria}, gold, source, language)


def _noul(state, instructions: str, gold: bool, source: str, language: str = "en", true: str | None = None,
          false: str | None = None) -> Example:
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true or false:
        q["criteria"] = {"true": true, "false": false}
    return _ex(state, q, "true" if gold else "false", source, language)


def _score(state, instructions: str, levels: list[str], gold: int, source: str, language: str = "en") -> Example:
    return _ex(state, {"type": "score", "instructions": instructions, "criteria": levels}, gold, source, language)


def derived_noul(choice: Example, rng: random.Random, template: str = "Is the correct answer: {option}?") -> Example:
    """A yes/no sibling of a choice question: asks about the gold option or a random wrong one."""
    names = list(choice["question"]["criteria"])
    option = choice["gold"] if rng.random() < 0.5 else rng.choice([n for n in names if n != choice["gold"]])
    return _noul(choice["state"], template.format(option=option), option == choice["gold"], choice["source"], choice["language"])


def subset_choice(choice: Example, rng: random.Random, k: int) -> Example:
    """The same question over a random subset of k options that keeps the gold one."""
    crit = choice["question"]["criteria"]
    pool = [n for n in crit if n != choice["gold"]]
    rng.shuffle(pool)
    keep = pool[:k - 1] + [choice["gold"]]
    rng.shuffle(keep)
    q = dict(choice["question"], criteria={n: crit[n] for n in keep})
    return dict(choice, question=q)


# ----------------------------------------------------------------------------- builders

NLI_LABELS = ["entailment", "neutral", "contradiction"]
NLI_CRITERIA = {"entailment": "The hypothesis follows from the premise",
                "neutral": "The hypothesis may or may not be true given the premise",
                "contradiction": "The hypothesis contradicts the premise"}
NLI_LEVELS = ["contradicted by the premise", "neither supported nor contradicted", "supported by the premise"]


def nli(name: str, language: str = "en") -> Builder:
    def build(r: dict, rng: random.Random) -> list[Example]:
        label = int(r["label"])
        if label < 0:
            return []
        state = {"premise": r["premise"], "hypothesis": r["hypothesis"]}
        return [
            _choice(state, "How does the hypothesis relate to the premise?", NLI_CRITERIA, NLI_LABELS[label], name, language),
            _noul(state, "Does the hypothesis follow from the premise?", label == 0, name, language,
                  "The premise entails the hypothesis", "It does not follow, or it contradicts the premise"),
            _score(state, "How strongly does the premise support the hypothesis?", NLI_LEVELS, 2 - label, name, language),
        ]
    return build


def qqp(r: dict, rng: random.Random) -> list[Example]:
    state = [r["question1"], r["question2"]] if rng.random() < 0.5 else {"first": r["question1"], "second": r["question2"]}
    same = int(r["label"]) == 1
    return [
        _noul(state, "Do these two questions ask the same thing?", same, "qqp"),
        _choice(state, "Are the two questions duplicates?", {"duplicate": "They ask for the same information",
                                                             "different": "They ask for different things"}, "duplicate" if same else "different", "qqp"),
    ]


def yelp_polarity(r: dict, rng: random.Random) -> list[Example]:
    pos = int(r["label"]) == 1
    return [
        _choice(r["text"], "Is this business review negative or positive?",
                {"negative": "The reviewer is dissatisfied", "positive": "The reviewer is satisfied"}, "positive" if pos else "negative", "yelp_polarity"),
        _noul(r["text"], "Would the reviewer recommend this business?", pos, "yelp_polarity"),
        _score(r["text"], "How satisfied is the reviewer?", ["dissatisfied", "satisfied"], int(pos), "yelp_polarity"),
    ]


DBPEDIA = ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation", "Building",
           "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"]


def dbpedia(r: dict, rng: random.Random) -> list[Example]:
    c = _choice({"title": r["title"], "content": r["content"]}, "Which kind of entity does this article describe?",
                {k: None for k in DBPEDIA}, DBPEDIA[int(r["label"])], "dbpedia_14")
    return [c, derived_noul(c, rng, "Does this article describe a {option}?")]


TREC_COARSE = {"ABBR": "Abbreviation", "ENTY": "Entity", "DESC": "Description or abstract concept", "HUM": "Human being",
               "LOC": "Location", "NUM": "Numeric value"}


def trec(r: dict, rng: random.Random) -> list[Example]:
    code = str(r.get("label_coarse_original") or r.get("label_original", "")).split(":")[0]
    if code not in TREC_COARSE:
        return []
    c = _choice(r["text"], "What kind of answer does this question ask for?", TREC_COARSE, code, "trec")
    return [c, derived_noul(c, rng, "Does this question ask for an answer of kind {option}?")]


def massive_builder(language: str) -> Builder:
    def build(r: dict, rng: random.Random) -> list[Example]:
        intents = r["_intent_names"]
        gold = str(r["label"])
        if gold not in intents:
            return []
        full = _choice(r["text"], "Which intent does this utterance express?", {i: i.replace("_", " ") for i in intents},
                       gold, f"massive_{language}", language)
        k = rng.choice([8, 12, 20, 30, 40])
        return [full, subset_choice(full, rng, k), derived_noul(full, rng, "Is the intent of this utterance {option}?")]
    return build


def hwu64(r: dict, rng: random.Random) -> list[Example]:
    intents = r["_label_names"]
    label = r.get("label", r.get("intent"))
    gold = intents[int(label)] if isinstance(label, int) or str(label).isdigit() else str(label)
    if gold not in intents:
        return []
    full = _choice(r.get("text") or r.get("utterance"), "Which assistant intent does this request express?",
                   {i: None for i in intents}, gold, "hwu64")
    return [full, subset_choice(full, rng, rng.choice([6, 10, 16, 24])), derived_noul(full, rng, "Is this request about {option}?")]


def _qa(state, texts: list[str], gold: str, source: str, instructions: str = "Which answer to the question is correct?") -> list[Example]:
    texts = list(dict.fromkeys(t.strip() for t in texts if t and t.strip()))
    if gold not in texts or len(texts) < 2:
        return []
    return [_choice(state, instructions, {t: None for t in texts}, gold, source)]


def openbookqa(r: dict, rng: random.Random) -> list[Example]:
    texts, labels = list(r["choices"]["text"]), list(r["choices"]["label"])
    gold = texts[labels.index(r["answerKey"])].strip()
    state = {"fact": r.get("fact1", ""), "question": r["question_stem"]} if r.get("fact1") else r["question_stem"]
    out = _qa(state, texts, gold, "openbookqa")
    return out + [derived_noul(out[0], rng)] if out else []


def piqa(r: dict, rng: random.Random) -> list[Example]:
    sols = [r["sol1"], r["sol2"]]
    out = _qa({"goal": r["goal"]}, sols, sols[int(r["label"])].strip(), "piqa", "Which solution achieves the goal?")
    return out + [derived_noul(out[0], rng, "Does this solution achieve the goal: {option}?")] if out else []


def social_iqa(r: dict, rng: random.Random) -> list[Example]:
    answers = [r["answerA"], r["answerB"], r["answerC"]]
    out = _qa({"context": r["context"], "question": r["question"]}, answers, answers[int(r["label"]) - 1].strip(),
              "social_iqa", "Which answer to the question fits the context best?")
    return out + [derived_noul(out[0], rng)] if out else []


def sciq(r: dict, rng: random.Random) -> list[Example]:
    options = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
    rng.shuffle(options)
    state = {"support": r["support"], "question": r["question"]} if r.get("support") else r["question"]
    out = _qa(state, options, r["correct_answer"].strip(), "sciq")
    return out + [derived_noul(out[0], rng)] if out else []


def race(r: dict, rng: random.Random) -> list[Example]:
    options = list(r["options"])
    out = _qa({"article": r["article"], "question": r["question"]}, options, options["ABCD".index(r["answer"])].strip(),
              "race", "Which option answers the question about the article?")
    return out + [derived_noul(out[0], rng)] if out else []


TOX_LEVELS = ["civil", "rude", "toxic", "severely toxic"]


def jigsaw(r: dict, rng: random.Random) -> list[Example]:
    toxic = int(r["toxic"]) == 1
    level = 3 if int(r.get("severe_toxic", 0)) == 1 else (2 if toxic else (1 if int(r.get("insult", 0)) or int(r.get("obscene", 0)) else 0))
    return [
        _noul(r["comment_text"], "Is this comment toxic?", toxic, "jigsaw",
              "Insulting, hateful, threatening or obscene", "Civil, even if critical"),
        _score(r["comment_text"], "How toxic is this comment?", TOX_LEVELS, level, "jigsaw"),
        _noul(r["comment_text"], "Does this comment contain a threat?", int(r.get("threat", 0)) == 1, "jigsaw"),
    ]


def ultrafeedback(r: dict, rng: random.Random) -> list[Example]:
    comps = r.get("completions") or []
    rated = [(c.get("response"), c.get("overall_score")) for c in comps if c.get("response") and c.get("overall_score") is not None]
    if len(rated) < 2:
        return []
    resp, score = rng.choice(rated)
    level10 = min(9, max(0, int(round(float(score))) - 1))  # overall_score is 1..10
    state = {"instruction": r["instruction"], "response": resp}
    return [
        _score(state, "How good is the assistant's response to the instruction?", [f"quality {i + 1} of 10" for i in range(10)], level10, "ultrafeedback"),
        _score(state, "How good is the assistant's response to the instruction?", ["very poor", "poor", "acceptable", "good", "excellent"], level10 // 2, "ultrafeedback"),
        _noul(state, "Is the response good enough to send to the user as is?", float(score) >= 7, "ultrafeedback"),
    ]


# ----------------------------------------------------------------------------- registry

def _src(name, dataset, config, split, licence, ok, build, qtypes, languages=("en",), needs=None) -> Source:
    return Source(name, dataset, config, split, licence, ok, build, qtypes, languages, needs=needs or {})


SOURCES: dict[str, Source] = {
    "multinli": _src("multinli", "nyu-mll/multi_nli", "default", "train", "CC BY-SA 3.0 / OANC", True, nli("multinli"), ("choice", "noul", "score")),
    "snli": _src("snli", "stanfordnlp/snli", "plain_text", "train", "CC BY-SA 4.0", True, nli("snli"), ("choice", "noul", "score")),
    "qqp": _src("qqp", "nyu-mll/glue", "qqp", "train", "Quora terms (unclear)", None, qqp, ("noul", "choice")),
    "yelp_polarity": _src("yelp_polarity", "fancyzhx/yelp_polarity", "plain_text", "train", "Yelp dataset terms (non-commercial research)", False,
                          yelp_polarity, ("choice", "noul", "score")),
    "dbpedia_14": _src("dbpedia_14", "fancyzhx/dbpedia_14", "dbpedia_14", "train", "CC BY-SA 3.0", True, dbpedia, ("choice", "noul")),
    "trec": _src("trec", "SetFit/TREC-QC", "default", "train", "unspecified (mirror of CogComp/trec)", None, trec, ("choice", "noul")),
    "hwu64": _src("hwu64", "DeepPavlov/hwu64", "default", "train", "CC BY 4.0", True, hwu64, ("choice", "noul"), needs={"_label_names": "rows:intents/intents:*"}),
    "openbookqa": _src("openbookqa", "allenai/openbookqa", "additional", "train", "Apache 2.0", True, openbookqa, ("choice", "noul")),
    "piqa": _src("piqa", "baber/piqa", "default", "train", "AFL 3.0 (mirror of ybisk/piqa)", True, piqa, ("choice", "noul")),
    "social_iqa": _src("social_iqa", "tasksource/social_i_qa", "default", "train", "CC BY 4.0 (mirror of allenai/social_i_qa)", True, social_iqa, ("choice", "noul")),
    "sciq": _src("sciq", "allenai/sciq", "default", "train", "CC BY-NC 3.0", False, sciq, ("choice", "noul")),
    "race": _src("race", "ehovy/race", "all", "train", "research use only (RACE terms)", False, race, ("choice", "noul")),
    "jigsaw": _src("jigsaw", "thesofakillers/jigsaw-toxic-comment-classification-challenge", "default", "train",
                   "CC0 1.0 (mirror of google/jigsaw_toxicity_pred)", True, jigsaw, ("noul", "score")),
    "ultrafeedback": _src("ultrafeedback", "openbmb/UltraFeedback", "default", "train", "MIT", True, ultrafeedback, ("score", "noul")),
}
for lang in ("de", "fr", "es", "ru", "ar", "hi", "zh"):
    SOURCES[f"xnli_{lang}"] = _src(f"xnli_{lang}", "facebook/xnli", lang, "train", "CC BY-NC 4.0", False, nli("xnli", lang),
                                   ("choice", "noul", "score"), (lang,))
for lang in ("en", "de", "fr", "es", "it", "pt", "ru", "ar", "hi", "zh", "ja"):
    SOURCES[f"massive_{lang}"] = _src(f"massive_{lang}", "mteb/amazon_massive_intent", {"zh": "zh-CN"}.get(lang, lang), "train",
                                      "CC BY 4.0 (mirror of AmazonScience/massive)", True, massive_builder(lang), ("choice", "noul"), (lang,),
                                      needs={"_intent_names": "collect:label"})
