"""Insurance claim triage with a JSON state and three typed questions in one pass."""
import json

from sysone.client import SysOneClient

claim = {
    "claim_id": "C-88213", "amount_usd": 4200, "policy_age_days": 11, "prior_claims": 3,
    "description": "Phone screen cracked when it fell off the kitchen counter. Also lost my laptop on the same day, "
                   "unrelated incident. No receipts, bought both for cash.",
}
res = SysOneClient().decide(claim, {
    "likely_fraud": {"type": "bool", "instructions": "Does this claim show strong indicators of fraud?"},
    "claim_type": {"type": "choice", "instructions": "Which claim type is this?",
                   "criteria": {"accidental_damage": "Item broken by accident", "theft": "Item stolen",
                                "loss": "Item lost or misplaced", "liability": "Damage to a third party"}},
    "severity": {"type": "score", "instructions": "How severe is this claim for the insurer?", "min": 1, "max": 5},
})
print(json.dumps(res, indent=2))
