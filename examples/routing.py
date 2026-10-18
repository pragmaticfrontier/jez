"""Ticket routing: act automatically when confident, escalate otherwise.

    .venv/bin/python examples/routing.py   (server must be running)
"""
from sysone.client import SysOneClient

client = SysOneClient()
QUESTIONS = {
    "routing": {"type": "choice", "instructions": "Which department should handle this ticket?",
                "criteria": {"billing": "Charges, invoices, and refunds",
                             "technical": "Software bugs and service outages",
                             "account": "Logins, passwords, and security"}},
    "is_urgent": {"type": "bool", "instructions": "Does the customer need a same-day response?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "min": 1, "max": 5,
                    "criteria": {"1": "calm", "3": "annoyed", "5": "furious, threatening to leave"}},
}
TICKETS = [
    "My card was charged twice for the monthly subscription.",
    "I can't log in since this morning, password reset email never arrives. Please help, I have a demo in 2 hours!",
    "The export button does nothing in Firefox, works in Chrome.",
    "hey quick one, where do i find my invoices?",
]
for t in TICKETS:
    res = client.decide(t, QUESTIONS)
    a = res["answers"]
    route, conf = a["routing"]["choice"], a["routing"]["confidence"]
    action = f"auto-route to {route}" if conf >= 0.9 else f"ESCALATE to human (best guess {route})"
    print(f"{res['latency_ms']:6.1f} ms | conf {conf:.2f} | urgent={a['is_urgent']['value']!s:5} "
          f"| frustration {a['frustration']['score']} | {action}\n    {t!r}")
