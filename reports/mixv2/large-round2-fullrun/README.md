# Jez-large round 2, full-epoch run (not released)

Gemma 4 E4B, LoRA r64/α128, lr 2e-4 chosen on in-mix validation, one full epoch of mix-v2 (Modal H100, 01:34–07:50 CEST
2026-10-18). In-mix validation 87.7% but Tier 1 mean 78.6 against 80.9 for the base (−2.4; 18 of 24 drops significant,
every Belebele language −3 to −8). Set aside on 2026-10-18 in favour of the anchored recipe pilot's step-1000 snapshot
(`reports/mixv2/pilots/README.md`), which keeps Gemma's general ability. Local weights: `releases/jez-large-1.0.0-round2-fullrun/`
(not in git); on the volume at `jez-releases:/mixv2/jez-large-1.0.0`.
