# War-room R13 SEED — System-2 → System-1 KNOWLEDGE TRANSFER (the forward report)

Owner mandate (asleep): produce ONE forward-looking report on S2→S1 knowledge transfer. Highlight any
ACTUALLY-POSITIVE transfer results we already have. The north star (owner's words, verbatim-ish):

> "Discovery becomes instinct which drives higher abstractions of discovery." A maneuver the System-2 VLM
> evokes in System-1 should become INSTINCTUAL/owned by S1 in that game, so S2 can stop naming it and move to
> higher-level abstractions (chain combos c1 c2 c3, not "press A then B"). Shared capabilities exist across
> games but are "subtly different enough that the broad concept holds but not the execution" → we need online
> ADDITION or TWEAKING, not offline LoRA ("nobody plays games like this"). Informally: close to OPTION LEARNING,
> options discovered by System-2. Do NOT overindex on the north star — game-specific LoRA is fine as a bootstrap
> to get past the single-level plateau; "sow seeds."

Architecture recap: frozen ~500M flow-matching DiT actor (S1, frame→18-action chunk, markov). Frozen VLM
(S2, emits text plan). Trainable bridge = Perceiver resampler + adapter + plan-head → K=8 continuous "plan
tokens" injected into DiT cross-attention. Masked-null = null plan reproduces base DiT EXACTLY (CFG-style plan
guidance). Qwen-LoRA on the VLM is now ALLOWED (owner relaxed the frozen-VLM constraint). Goal: ONE generalist
across all games (owner REJECTS per-game/per-genre models; intra-genre interference is a problem to SOLVE).

## THE POSITIVE TRANSFER RESULTS WE ALREADY HAVE (ground every claim in these)

P1. **EVOCATION WORKS (S2 can summon a primitive S1 already owns).** SMW duck-on-command: BASE bridge DOWN-press
    1.2% (terse plan) → 3.4% ("duck to dodge") → 10.8% ("press down to duck") = 9× plan-steerable. The DiT HAS
    duck; the plan TOKEN evokes it. (files/DISCRIMINATOR_RESULTS.md "DUCK-ON-COMMAND".)
P2. **ADDITION NEEDED for precise OUTCOMES (S1 doesn't own it; naming isn't enough).** spin-jump-kills-Rex:
    JUMP button 7.2× evocable by plan, BUT the kill/dispatch OUTCOME (survive+advance past Rex) flat 3.75%→7.5%
    (both near chance). The maneuver is an ADDITION at the outcome level even though the jump primitive is
    evocable → route to demo-fit/DiT-LoRA on human rex-kill chunks (long path). (maneuver_router, R11.)
    => CLEAN DICHOTOMY: evocation (free, S2 names a skill S1 has) vs addition (costly, S1 must LEARN the skill).
P3. **POSITIVE CROSS-GAME TRANSFER at the data level.** ONE pooled plan-head fit on SMW+MMX+SMB1 (each chunk on
    ITS game's plan) widens Δ_plan on ALL (SMW +26, MMX +31, 3/3 seeds, CIs>0, null-invariant 0.0). SMW BENEFITS
    from pooling (1.3× vs solo): SMB1+MMX demos HELP SMW. Generalizes to a NEW game (MMX) out of the box (+117%).
P4. **NO-INTERFERENCE MERGE (skills compose).** pooled plan-head (plan_head.*) + Sonic RWBC (lora_*) = mostly-
    disjoint params → merged on one btn_s600, all 4 games retained/improved, null-invariance exact. Two modes
    write different param groups → stack cleanly. (CAPSTONE + full_merge_eval.)
P5. **CONSOLIDATION WITHOUT FORGETTING is achievable (KL-anchor).** Naive demo-fit on ONE terse plan CATASTROPH-
    ICALLY overwrote the bridge's plan-responsiveness (duck 0% for ALL plans — broke evocation). KL-anchor (L2 of
    plan-tokens to a frozen pre-fit head over a broad plan dist) ADDS the advance skill AND PRESERVES the whole
    action space (expr-battery R=0.83, only PASS; Δ_plan +45.1, 3/3). = CoTTA stochastic-restoration in
    functional form. This is the "add a skill without destroying the others" primitive. (R9 FINAL.)
P6. **STALENESS IS TEXT-BOUND; the continuous short path is REMARKABLY STABLE.** K plan tokens barely change
    frame-to-frame (tok-drift ~0.001; cached==fresh==null << oracle). The owner's "50-60% plan flip" is in the
    TEXT, not the tokens. Re-grounding tokens (forward-pass TTA, von Oswald "short path is already TTT") does NOT
    help — no token-staleness to fix. LIVE re-planning HURTS (MMX live-fresh −11) — churns text without payoff.
    Lever = plan TEXT QUALITY (VLM-abstraction / Qwen-LoRA), not token-TTA, not actor-weight-TTT. (R11.)
P7. **demo-fit FIXES the live-deployment gap.** Real deployment uses the live VLM plan, not the oracle. On BASE
    the live plan HURTS SMW (−10.7); demo-fit → +56.3 (≈ oracle). Makes the bridge robust to imperfect S2 plans.

## THE FORWARD QUESTIONS (answer with mechanisms + the next decisive experiment, ground in P1–P7 + lit)

Q1. **Evocation/Addition as THE transfer taxonomy.** P1 vs P2 give us a measured dichotomy. Formalize a
    DECISION RULE: given a skill the owner wants in game G, how do we cheaply CLASSIFY it as evocable (free,
    S2-names) vs addition (needs S1 learning)? (We have maneuver_router: null-AUC reflex + evoc-ratio + emulator
    survive+advance outcome.) What's the minimal online test? This routes effort.

Q2. **PERSISTENT EVOCATION → INSTINCT (the core north-star mechanism).** A skill S2 evokes transiently (P1)
    should become OWNED by S1 so S2 can stop naming it (free up plan bandwidth for higher abstractions). Candidate
    mechanisms — rank them, name the one to try first, give a falsifier:
    (a) AMORTIZED CONSOLIDATION: periodically demo-fit/distill the EVOKED behavior into the policy (KL-anchored,
        P5) so it fires WITHOUT the plan token → frees the token. (Is this just self-distillation of the
        plan-conditioned policy into the null policy on states where the skill should fire?)
    (b) FAST-WEIGHTS / hypernetwork: S2 writes a transient weight delta per game/episode (online tweak, P6's
        "text-bound" suggests tokens are stable so maybe weights are the right online surface for EXECUTION
        nuance). vs offline LoRA ("nobody plays games like this").
    (c) SKILL-TOKEN CODEBOOK: a compact learned vocabulary of options-as-tokens (not prose) that S2 emits and S1
        expands — addresses the token-dilution wall (Q3). Discovery (S2 finds a useful chunk) → mint a token →
        S1 learns to expand it → S2 reuses it at higher abstraction. This is literally "options discovered by S2".
    Which of (a)/(b)/(c) does P5 (KL-anchor add-without-forget) most enable? Is consolidation (a) the bridge from
    transient evocation to owned instinct, with the skill-token (c) as the eventual compression?

Q3. **TOKEN-DILUTION WALL.** We can't verbose-prompt every primitive (recall image-token-overwhelm EXP-050/051;
    P6 shows live text churn hurts). As games recurse in complexity S2 must say MORE in the SAME K=8 budget.
    Options: widen K (8→32, bounded), or a compact skill-token codebook (Q2c), or hierarchical plans (S2 emits
    abstract options, a mid-layer expands). Given P6 (tokens stable + load-bearing) and the masked-null
    constraint, what's the bandwidth path that PRESERVES null-invariance? Does a codebook of skill-tokens (each a
    learned K-token macro) compress better than prose?

Q4. **SUBTLY-DIFFERENT EXECUTION (the generalization nuance).** Owner: "broad concept holds, execution differs
    per game." Spin-jump exists in SMW but the timing/sprite-interaction differs from a generic jump. P3/P4 show
    CONCEPT transfers (pooled, cross-game), but P2 shows the precise OUTCOME does not (Rex). So: concept = shared
    plan-token direction; execution = per-game motor detail that needs a small ADD. Is the right architecture a
    SHARED concept channel (the generalist plan-head, transfers) + a small per-game/per-state EXECUTION adapter
    (fast-weight or gated DiT-LoRA, the addition) that the router (Q1) gates on? How to keep it ONE model (owner
    rejects per-game) — e.g. a single hypernet/codebook indexed by game-context, not separate weights?

Q5. **THE RECURSION ("discovery → instinct → higher discovery").** Sketch the loop concretely with OUR tools:
    S2 proposes a maneuver → emulator save-state verifies it (P2's survive+advance outcome = dense checkable
    reward) → if it works, CONSOLIDATE it into S1 (P5 KL-anchor) and/or mint a skill-token (Q2c) → S2 now treats
    it as a primitive and composes at a higher level (chain combos) → repeat. This is Expert-Iteration / VOYAGER
    skill-library / option-discovery. What is the MINIMAL version we can demonstrate THIS WEEK on SMW (e.g. evoke
    duck → consolidate so null-policy ducks at bullet states → S2 stops naming duck)? Give the experiment.

Q6. **HIGHLIGHT REEL.** Which of P1–P7 is the strongest "positive S2→S1 transfer" result to lead the report
    with, and how should we frame it honestly (what it proves vs what it assumes — most evals are oracle-plan,
    short-horizon, fixed save-states; closed-loop long-horizon drift is unproven)?

## Constraints / falsifier discipline
- ONE generalist (no per-game models). Null-invariance EXACT (masked-null). VLM frozen-first but Qwen-LoRA OK.
- Every proposed mechanism needs a NAMED FALSIFIER and a minimal experiment using EXISTING tools (demo_bc.py
  --kl-anchor, maneuver_router.py, staleness_probe.py, emulator save-states, pooled_demofit.py).
- Don't re-litigate settled findings (staleness text-bound P6; RWBC-for-plan-OOD-collapses; evocation works P1).
- Lit must be arXiv-grounded (options/HRL Sutton-Precup-Singh; VOYAGER skill library; fast-weights/hypernetworks
  Ha/Schmidhuber; policy distillation; RT-H language-motions; in-context skill acquisition; DIAYN; option-critic).

DELIVERABLE from each agent: a position memo answering Q1–Q6, ranking mechanisms for Q2 (the core), naming the
ONE minimal this-week experiment (Q5), and the highlight framing (Q6). Cite P1–P7 and arXiv lit.
