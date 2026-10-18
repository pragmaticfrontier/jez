"""Evaluation of Jez against published decision-model figures.

* :mod:`eval.data`      row loading from the Hugging Face datasets-server with a JSONL cache
* :mod:`eval.tasks`     the Tier 1 datasets mapped onto Jev questions, plus Tier 2 local loaders
* :mod:`eval.run_eval`  score a backend on sampled or full splits; accuracy, ECE, Brier, latency
* :mod:`eval.compare`   set Jez beside Jev, Microsoft-Decision-1, the Decisions API, Laya and the base model
"""
