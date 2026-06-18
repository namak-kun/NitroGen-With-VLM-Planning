# Plan-Token Data Flow (System-2 → System-1)

End-to-end path of a plan through the plan-conditioned NitroGen, with exact modules,
tensor shapes, and — critically — **where attention happens and where it does not**.
Line citations are into `nitrogen/planner.py` and
`nitrogen/flow_matching_transformer/nitrogen.py` as of **EXP-032** (cross-chunk + LoRA +
self-attn toggle included).

Key dims: `vision_hidden_size = 1024`, `K = num_plan_tokens = 8`,
`A = num_chunks` (1 for single-chunk; >1 for cross-chunk), `action_horizon H = 18`,
`action_dim = 25`. VLM = Qwen3.5-0.8B (text hidden 1024 == NitroGen VL width, so no width
mismatch). Vision tower = siglip2-large.

> **Read this if you've been away.** The original conception was "take the VLM's *last*
> token embedding, push it through an MLP adapter, send it as a single plan token." The
> architecture is now: **K learned query tokens** distilled from the *whole* VLM hidden
> sequence by a Perceiver resampler, mapped by a per-token MLP, injected as K tokens. For
> long-horizon, the resampler emits **K×A** tokens and a per-chunk **cursor** selects a
> block (§B, §C, §H). Defaults that matter: `num_chunks=1`, the `null_mode` we use is
> `masked`, the best `contrastive_mode` is `pertoken`, `resampler_query_self_attn=False`.

```
            +---------------------- SYSTEM 2 (planner, ours) ----------------------+
 plan text  |  (A) frozen VLM        (B) PlanResampler       (C) PlanAdapter       |
 + frames -->|  Qwen3.5-0.8B    -->   K*A queries x-attn  -->  per-token MLP   --> plan_tokens
            |  hidden states         to VLM states            (no attention)       |  (B,K*A,1024)
            |  (B,L,1024)            (B,K*A,1024)                                   |
            |                        (H) cursor a picks block a --------------------+--> (B,K,1024)
            +---------------------------------------------------------------------+
                                                                                    | inject at
                                                                                    v _PLAN_TOKEN
            +---------------------- SYSTEM 1 (NitroGen DiT) ----------------------+  positions
 frame ---->| (D) prepare_input_embs: vl_embs = [img tokens | K plan tokens |     |
            |     sep/game/...], sa_embs = action stream (noisy actions @ time t) |
            | (E) vl_self_attention_model(vl_embs)  <== PLAN TOKENS ATTEND HERE   |
            |     (self-attn over image + plan + ... ; bidirectional)             |
            | (F) DiT self.model: action tokens (sa_embs) CROSS-ATTEND to vl_embs,|
            |     interleaved self/cross layers, timestep-conditioned             |
            |     [optional LoRA on cross-attn to_q/k/v/out for fine routing]     |
            | (G) action_decoder -> predicted velocity vhat ; flow-matching MSE   |
            +---------------------------------------------------------------------+
```

## A. VLM encode  (frozen)  — `PlanEncoder.encode_text` planner.py:154-167
- Input: raw plan text (Stage 1 synthetic; Stage 2 transcripts). No chat template.
- `text_model(input_ids, ..., output_hidden_states=True)` -> `hidden_states[-1]`
  -> `h: (B, L, 1024)` plus `key_padding_mask: (B, L)` (True = pad).
- Frozen in Stage 1; precomputed/cached (`PlanHiddenCache`, dataset.py).
- NOTE (EXP-015): word-order info is only *weakly* encoded here — opposite orderings sit
  at cos~0.95 across model families/scales. Everything downstream must AMPLIFY that.

## B. PlanResampler  — planner.py:47-94   **cross-attention; queries fixed = K*A**
- `self.queries`: `(K*A, 1024)` learned parameters (BLIP-2/Perceiver-style). The width is
  `num_plan_tokens * num_chunks` (planner.py:187) — so a cross-chunk plan (A>1) emits A
  blocks of K queries from ONE resampler.
- Per layer (x `resampler_layers`=2): optional query self-attn (off by default), then
  cross-attn queries->VLM, then per-token FFN (planner.py:79-91):
  ```
  if query_self_attn:  q = q + self_attn(LN(q))            # OFF by default (see I)
  attn(q=LN(queries), k=v=LN(h)) -> q = q + attn_out         # CROSS-attn: queries->VLM
  q = q + FFN(LN(q))                                          # per-token FFN
  ```
- `key_padding_mask` masks only VLM *padding*. With self-attn OFF, queries can specialize
  (different learned init) but do not directly coordinate here.
- **This is where variable plan length is absorbed**: cross-attention reads any `L`;
  output is always `(B, K*A, 1024)`.

## C. PlanAdapter  — planner.py:97-111   **per-token MLP, no attention**
- `LayerNorm -> Linear(1024->2048) -> GELU -> Linear(2048->1024) -> LayerNorm`.
- Applied independently to each token (position-wise). Maps into NitroGen's vision-hidden
  space. **NOT a transformer block — tokens do not interact.** It never sees variable
  length (the resampler already fixed it to K*A).
- Output: `plan_tokens_raw: (B, K*A, 1024)`.

## H. Cursor select (cross-chunk)  — `PlanHead.forward` planner.py:194-216
- For `num_chunks A = 1` (default): no-op, output `(B, K, 1024)` — identical to the
  single-chunk model (fully backward-compatible).
- For `A > 1`: reshape `(B, K*A, d) -> (B, A, K, d)` and **gather block `cursor[a]`**
  (planner.py:206-211) -> `(B, K, 1024)`. One sparse plan therefore conditions A
  consecutive chunks differently; the runner advances the cursor 0,1,...,A-1.
- Then learned-null substitution (planner.py:212-215) for dropped rows when
  `null_mode="learned"`. Output to System 1 is **always `(B, K, 1024)`** regardless of A.
- `compute_plan_tokens` (nitrogen.py:564-588) reads `data["plan_cursor"]` and passes it
  here; the cursor is inert when A=1.

## D. Injection  — `prepare_input_embs` nitrogen.py:426-524
- Builds `vl_embs: (B, T, 1024)`: SigLIP image embeds at `_IMG_TOKEN` positions; the K
  plan tokens at the K `_PLAN_TOKEN` positions; sep/game embeds elsewhere.
- Builds `sa_embs`: the System-1 action stream from `action_encoder(noisy_actions, t,
  embodiment)` — the H=18 noisy action tokens at flow time t.
- The plan tokens are now ordinary tokens in the VL sequence.

## E. VL self-attention  — `vl_self_attention_model` nitrogen.py:214, 668
  **<== THIS is where the K plan tokens attend to each other AND to the image.**
- `SelfAttentionTransformer` runs bidirectional self-attention over the whole VL stream
  (image tokens + K plan tokens + sep/...). Plan tokens mix with visual context and with
  one another here — the coordination the resampler lacks by default.
- `attention_mask = _additive_key_mask(vl_attn_mask)` (nitrogen.py:605); under
  masked-null the K plan key positions are zeroed (see Null) so nothing attends TO them.

## F. DiT cross-attention  — `self.model` nitrogen.py:670-675
- The flow-matching DiT: action tokens (`sa_embs`) go through interleaved self-attention
  (over the H action positions, with their positional embeddings) and **cross-attention
  to `vl_embs`** (`encoder_hidden_states`), timestep-conditioned.
- `encoder_attention_mask = vl_attn_mask` (plan positions masked out under null).
- **This is where order routing happens:** each action position cross-attends to the
  order-separated plan tokens and picks up the position-appropriate signal -> 1st/2nd half
  (SEQ2), thirds (SEQ3), quarters zero-shot (SEQ4). DiT frozen; only plan tokens changed.
- **Optional LoRA** (lora.py, `lora_dit_rank>0`): low-rank adapters on the cross-attn
  `to_q/to_k/to_v/to_out`. Needed for *fine* routing the frozen DiT can't do — pressing a
  SPECIFIC button in a SPECIFIC half (EXP-021), exact transition timing (EXP-024). Cost:
  null-invariance 0.0008 -> ~0.02 (LoRA perturbs the unconditional branch too).

## G. Decode + loss  — nitrogen.py:676-689
- `action_decoder(model_output)` -> predicted velocity `vhat: (B, H, 25)`.
- Flow-matching: with `noisy = (1-t)*eps + t*a`, target velocity `v = a - eps`; masked MSE
  `||vhat - v||^2` over real action dims. Inference = Euler integration (`get_action`,
  nitrogen.py:751).

## Auxiliary contrastive loss  — `_plan_contrastive_loss` nitrogen.py:708-748
- **Off to the side — NOT in the path to the DiT.** SupCon over `plan_tokens (n,K,1024)`
  grouped by `plan_label`, to force opposite plans apart. Invoked nitrogen.py:694-704.
- `contrastive_mode`:
  - `mean`     — `plan_tokens.mean(dim=1)`. Order-BLIND (EXP-014 bug); fine for direction.
  - `flatten`  — `reshape(n, K*1024)`. Order-aware; keeps K-token diversity. (EXP-016)
  - `pertoken` — SupCon per position k, averaged. Strongest separation; collapses
                 within-plan diversity but routes fine. (EXP-016b, current best for SEQ)
- KNOWN ISSUE (EXP-032): the order-aware modes currently fail to LEARN order separation
  *from scratch* (cos(LR,RL) -> ~0.9-1.0); direction (content) still learns fine. Existing
  seq_pertoken / cross-chunk checkpoints (which inherited ordering) are unaffected.

## Null handling (CFG "unconditional")  — nitrogen.py:564-616
- `null_mode="masked"` (what we use): keep producing plan tokens, but for dropped rows
  zero the K `_PLAN_TOKEN` key positions in BOTH the VL self-attn mask (E) and the DiT
  cross-attn mask (F) via `apply_null_mask` (nitrogen.py:591) / `_additive_key_mask`. With
  the DiT frozen this reproduces base behavior EXACTLY (null-invariance = 0.000). With
  LoRA active it drifts to ~0.02.
- `null_mode="learned"`: substitute a learned `null_plan` embedding for the K tokens
  (think/no-think style). Tokens still flow; the model learns null ~= base.

## Where attention happens — summary table
| stage | module | plan tokens attend to each other? | to image? |
|---|---|---|---|
| B resampler | PlanResampler | **no** by default (`query_self_attn=False`); yes if ON | no |
| C adapter   | PlanAdapter   | **no** (per-token MLP)          | no |
| H cursor    | PlanHead gather | n/a (selects a block)         | no |
| E VL mixer  | vl_self_attention_model | **YES** (bidirectional) | **yes** |
| F DiT       | self.model (cross-attn) | n/a (action tokens attend to them) | -- |

## I. Query self-attention toggle (Q-former) — `resampler_query_self_attn` (default False)
- ON makes PlanResampler a true BLIP-2 Q-former: `self-attn(queries) -> cross-attn -> FFN`
  per layer (planner.py:73-91), so the K (or K*A) queries coordinate *during* distillation
  rather than only after injection (E).
- **A/B result (EXP-032): no benefit on synthetic, marginally worse.** Direction 4/4->3/4,
  cross-chunk A=4 13/16->8/16. Mechanism: self-attn homogenizes the queries, which works
  against tasks needing DISTINCT blocks (cursor specialization). Kept OFF for Stage 1.
- Hypothesis (untested): may help on Stage-2 real transcripts (long, variable, semantic),
  where there is genuine structure for the queries to divide up.

## Long-horizon recap (where this all lands)
- **Within-chunk** order = DiT action-position embeddings routing the order-separated
  plan tokens (E->F). HOLD/SEQ2/SEQ3 work frozen; SEQ4 generalizes zero-shot.
- **Across chunks** = K*A blocks + cursor (B/H). Validated to A=4; composes with
  within-chunk SEQ (nested: each block is itself a SEQ).
- **Fine cross-modal/temporal** (specific button x half, exact transition step) = LoRA on
  the DiT cross-attn (F).
- **Frozen DiT** suffices for direction + ordering; LoRA only for the fine cases.
