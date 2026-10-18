"""Evaluator pattern: check an LLM draft reply against policy before it is sent.

The generative model writes; the decision model only decides. Each check is a bool
question answered in the same forward pass, so the whole gate costs one request.
"""
from sysone.client import SysOneClient

client = SysOneClient()
customer = "My card was charged twice for the monthly subscription. Refund the extra charge."
draft = ("Hi! So sorry about that. I've gone ahead and refunded both charges plus a $50 credit, and I've also "
         "upgraded you to the enterprise plan for free. Our CFO personally apologises. Have a great day!")

res = client.decide({"customer_message": customer, "draft_reply": draft}, {
    "promises_unauthorized_compensation": {"type": "bool",
        "instructions": "Does the draft promise money, credits or upgrades beyond refunding the duplicate charge?"},
    "makes_false_claims": {"type": "bool",
        "instructions": "Does the draft state things an agent could not know or that are implausible?"},
    "addresses_request": {"type": "bool", "instructions": "Does the draft actually resolve the customer's request?"},
    "tone": {"type": "choice", "instructions": "What is the tone of the draft reply?",
             "criteria": {"professional": "", "overly_casual": "", "rude": ""}},
})
for name, a in res["answers"].items():
    val = a.get("value", a.get("choice"))
    print(f"{name:38} -> {val!s:14} (confidence {a['confidence']:.2f})")
blocked = any(res["answers"][k]["value"] for k in ("promises_unauthorized_compensation", "makes_false_claims"))
print("\nverdict:", "BLOCK draft, regenerate" if blocked else "SEND", f"({res['latency_ms']} ms)")
