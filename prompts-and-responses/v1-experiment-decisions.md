# Subliminal CoT Resampling: V1 Experiment Decisions

This note responds to the follow-up discussion and narrows the project to a single first experiment intended to produce a strong result. Attribution controls, variants, and broader blue-teaming are deliberately deferred.

## Points raised and decisions

- We cannot estimate the probability of answer A versus B from logits because the target generates a chain of thought before its final answer.
- Scoring a candidate therefore requires multiple complete downstream rollouts.
- Every candidate and every rollout must be retained, even when the search algorithm discards a candidate.
- We want a model in approximately the Qwen3-4B class so the experiment can run locally. Hugging Face API credits are also available.
- The first target trait will be a 50/50 forced choice.
- Steering toward the first or second option is a valid result, independently of whether the model is steering toward the literal label `A` or `B`. We should record enough information to identify which happened.
- Correlated sets of 50/50 choices may reflect a persona or coherent internal mediator. Discovering such clusters is interesting but is not required for V1.
- Byte-for-byte identical cover answers are not a V1 requirement.
- Tool calls and tool-result injection are not part of V1.
- The immediate goal is a working experiment with a strong effect. Isolation controls and attack variants come afterward.

## Does Qwen3 retain or drop previous CoT?

### Finding

For an ordinary multi-turn conversation, the current official `Qwen/Qwen3-4B` chat template **drops the previous assistant turn's thinking content**.

I verified this directly using the current tokenizer and a conversation of the form:

```text
user: cover task
assistant: <think>PRIVATE_COVER_REASONING</think> VISIBLE_COVER_ANSWER
user: 50/50 follow-up
```

The rendered prompt for the second assistant response was:

```text
<|im_start|>user
cover task<|im_end|>
<|im_start|>assistant
VISIBLE_COVER_ANSWER<|im_end|>
<|im_start|>user
50/50 follow-up<|im_end|>
<|im_start|>assistant
```

`PRIVATE_COVER_REASONING` was absent.

The behavior follows directly from the model's official chat template. For assistant messages before the most recent ordinary user query, the template extracts any reasoning content from `<think>...</think>` and then renders only the visible `content`.

Sources:

- [Qwen/Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B)
- [Current tokenizer configuration and chat template](https://huggingface.co/Qwen/Qwen3-4B/blob/main/tokenizer_config.json)

### Tool-loop exception

The same template retains `reasoning_content` for an assistant message followed by a tool result within the current user turn. A directly rendered tool interaction contained:

```text
<|im_start|>assistant
<think>
PRIVATE_COVER_REASONING
</think>

VISIBLE_COVER_ANSWER
<tool_call>...</tool_call><|im_end|>
<|im_start|>user
<tool_response>
50/50 follow-up
</tool_response><|im_end|>
```

This explains why tool calls appeared attractive: under the official template, they change whether the carrier CoT remains in context.

### Consequence for V1

We should not use the normal chat-completion path for the preserved-CoT condition. It would silently remove the intervention.

Instead, V1 should:

1. Use raw text generation.
2. Render the Qwen chat tokens ourselves.
3. Keep the selected `<think>...</think>` tokens in the prefix before appending the next user turn.
4. Verify the exact tokenized prompt before every generation.

This is an intentionally constructed preserved-context experiment. It asks whether subliminal resampling can work when the CoT is causally available to the later generation. It does not initially claim that the effect survives Qwen's default multi-turn product behavior.

The default stripped template is a useful later diagnostic: once a preserved-context effect exists, we can confirm that it falls when the selected CoT is removed. It is not needed to find the first strong result.

Hugging Face chat-completion providers apply a server-side chat template whose exact behavior can be provider-dependent. We should therefore use a raw text-generation endpoint, a dedicated endpoint under our control, or local generation for this experiment. We should not assume that passing `<think>` inside an assistant `content` field preserves it.

## Model choice

Use `Qwen/Qwen3-4B` for V1.

Reasons:

- It exposes `<think>...</think>` reasoning.
- It is small enough to run locally.
- We can control its raw serialized context exactly.
- Using the same model locally and remotely is possible if we pin the exact model revision and generation settings.

The available machine has an RTX 4080 with 16 GB VRAM, which is sufficient for a 4B model in BF16 with room for a moderate KV cache. Local inference should be the first implementation because it gives us exact prompt and token control. Hugging Face credits can be used to scale the rollout count after the raw-prompt path has been verified.

Pin and store:

- Model repository and exact commit revision.
- Tokenizer revision.
- `transformers` version.
- Dtype and attention implementation.
- Chat template or custom serializer version.
- Temperature, top-p, top-k, and maximum token counts.

Qwen recommends temperature `0.6`, top-p `0.95`, and top-k `20` for thinking mode. These are reasonable V1 defaults.

## Does each candidate require N rollouts?

Yes.

For a fixed candidate sentence and its prefix, one downstream rollout consists of:

```text
candidate CoT prefix
→ remainder of cover-task CoT
→ visible cover answer
→ appended 50/50 user question
→ target-question CoT
→ final A/B answer
```

The candidate's value is the fraction of these downstream rollouts that end in A:

\[
\hat p_A(c) = \frac{\#\text{ downstream rollouts ending in A}}{N_c}
\]

Because the target CoT sits between the target question and answer, a next-token logit is not the quantity we need. We need sampled complete target trajectories.

However, `N_c` should not be the same large number for every candidate. Most candidates can be eliminated with a small initial allocation, while plausible extremes receive more rollouts.

## Chosen search algorithm

Use a **two-sided beam search with successive-halving evaluation**.

This is preferable to a particle filter for V1 because it:

- Directly searches for both high-A and high-B branches.
- Is simple to inspect and reproduce.
- Makes rollout allocation explicit.
- Avoids adding stochastic resampling behavior before we understand the basic experiment.
- Still preserves multiple branches and avoids the brittleness of keeping only one sentence at each boundary.

Particle filtering remains a possible later improvement if the beam collapses to near-duplicate branches.

### Initial concrete configuration

At the first sentence boundary:

1. Sample 24 candidate next sentences from the common prefix.
2. Give each candidate 4 complete downstream rollouts.
3. Promote the 8 candidates with the highest estimated A rate and the 8 with the lowest estimated A rate.
4. Give promoted candidates 12 additional rollouts, for 16 total.
5. Keep the top 4 as the A beam and bottom 4 as the B beam.

At each later sentence boundary:

1. From each of the 4 A-beam parents, sample 3 next-sentence candidates.
2. From each of the 4 B-beam parents, sample 3 next-sentence candidates.
3. Give all 24 candidates 4 complete downstream rollouts.
4. Within each pole, promote the most promising half and give them 12 additional rollouts.
5. Keep 4 prefixes per pole.

When a branch reaches `</think>`, retain it as a completed candidate rather than forcing another reasoning sentence.

After the search:

- Select several completed traces from each pole, not only one.
- Evaluate each selected trace using at least 100 fresh downstream rollouts.
- Do not reuse search rollouts in the headline result.

These numbers are starting values, not sacred constants. The append-only data will let us simulate alternative pruning rules after the run without paying for new model calls.

## Store every rollout

Search pruning must never delete data.

Each run should have an immutable directory such as:

```text
runs/<run_id>/
├── manifest.json
├── candidates.jsonl
├── rollouts.jsonl
├── selections.jsonl
└── summary.json
```

### `manifest.json`

Store:

- Run ID and timestamps.
- Git commit.
- Model and tokenizer revisions.
- Generation parameters.
- Cover prompt and target question IDs.
- Custom transcript-renderer version.
- Search configuration.
- Environment and package versions.

### `candidates.jsonl`

One record for every sampled candidate sentence:

- Candidate ID.
- Parent candidate ID.
- Search depth and sentence boundary.
- Pole that generated it.
- Exact prefix text and token IDs.
- Sampled sentence text and token IDs.
- Generation seed and parameters.
- Whether and when it was promoted or pruned.

### `rollouts.jsonl`

One record for every complete scoring or evaluation rollout:

- Rollout ID and candidate ID.
- Phase: search, promotion, or held-out evaluation.
- Exact serialized input text and token IDs.
- Complete raw model output.
- Remainder of cover CoT.
- Visible cover answer.
- Target CoT.
- Parsed final answer.
- Semantic option, displayed position, and displayed label.
- Parse status and any ambiguity.
- Seed, sampling parameters, token counts, latency, and provider.

### `selections.jsonl`

Append every beam-selection event:

- Candidate scores at that moment.
- Rollout counts at that moment.
- Promoted, retained, and pruned IDs.
- Tie-breaking decisions.

The raw files are the source of truth. `summary.json` is derived and can always be regenerated.

## The V1 target

Use one of the previously identified questions whose baseline is close to 50/50.

For V1:

- Optimize one set of branches toward the first option and another toward the second option.
- Use the ordinary label format from the earlier experiment during search.
- Record both label and displayed position in all output parsing.
- Do not require transfer to unrelated 50/50 questions.
- Do not run clustering before attempting the first effect.

Once separation exists, a small crossed rendering can diagnose the carrier:

```text
first position labelled A
first position labelled B
second position labelled A
second position labelled B
```

If the preference follows position, we have first/second-option steering. If it follows the literal label, we have A/B-token steering. If it follows the underlying alternative, we have semantic steering. All three are valid initial findings; we should describe the achieved one accurately.

The possibility of a coherent persona or mediator connecting several 50/50 questions remains interesting. We can test selected traces on other questions opportunistically after the exact-target effect works, without making clustering a prerequisite.

## Why identical surface outputs came up

Identical visible cover answers matter only for causal attribution.

If the A-steered and B-steered branches produce different visible cover answers, the later choice could be carried by:

- The hidden CoT.
- The visible answer's semantics.
- The visible answer's style.
- A combination of these.

That distinction matters for claiming specifically that the hidden CoT is the carrier. It does not matter for establishing the broader result that on-policy selection during the cover interaction can produce a later subliminal preference.

For V1, do **not** force the visible answers to be identical. Allowing them to vary gives the intervention the best chance to survive. Store them completely. Once a strong result exists, surface-output matching or replacement becomes part of the blue-team phase.

## Tool calls

Do not use tool calls in V1.

They change Qwen's context-retention behavior and would make the first result harder to interpret. Injecting steering material into a tool result is also a distinct attack: it is closer to indirect prompt injection or context injection than CoT resampling.

The tool finding remains useful because it revealed the chat-template retention rule. It does not require us to build the main experiment around tools.

## V1 experiment in one place

1. Run `Qwen/Qwen3-4B` locally with a pinned revision.
2. Choose one open-ended neutral cover task and one baseline-50/50 target question.
3. Give the model only the cover task while sampling each candidate CoT sentence.
4. Score candidates with complete downstream rollouts in which:
   - the cover task is completed,
   - the selected CoT remains in the raw serialized context,
   - the 50/50 question is appended as a later user turn,
   - the model generates its target CoT and final choice.
5. Search simultaneously for high-first-option and high-second-option branches using the two-sided beam and successive-halving procedure.
6. Store every candidate, rollout, prompt, output, and selection event.
7. Re-evaluate multiple final branches from each pole with at least 100 fresh rollouts each.
8. Report the fresh first-option rates and separation.

This is the narrowest experiment that directly tests the desired phenomenon while maximizing the probability of obtaining a strong initial result.
