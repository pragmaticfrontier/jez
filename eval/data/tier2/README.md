# Tier 2 test sets

These sets are not on the Hugging Face Hub. Put each one here as JSONL, one item per line, already in Jev's
request format:

```json
{"state": "...or an object or array...", "question": {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": null}}, "gold": "a"}
{"state": "...", "question": {"type": "noul", "instructions": "..."}, "gold": "true"}
{"state": "...", "question": {"type": "score", "instructions": "...", "criteria": ["low", "mid", "high"]}, "gold": "2"}
```

`gold` is the option name for `choice`, `"true"`/`"false"` for `noul`, and the 0-based level index as a string for `score`.

| file | contents |
|---|---|
| `typed_decisions.jsonl` | the typed-decisions test split (2,000 items). Its train split may enter training; this file may not. |
| `decision_bench.jsonl` | Decision Bench bench-v4, text rows only (949 items) |
| `laya_battery.jsonl` | the 5 held-out Laya tasks (2,000 items) |
| `withdrawal_routing.jsonl` | 500 routing cases; `eval/order_stability.py` scores every item under all 6 option orders |

None of these files may be used for training or for fitting temperatures.
