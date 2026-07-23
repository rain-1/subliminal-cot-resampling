# Subliminal Chain-of-Thought Resampling: Experimental Proposals

I think this can work, but the best version is not “create a biased CoT and hope the bias survives.” It is to optimize directly for bias that survives the cover task.

## Proposed core experiment

The chronology should be:

```text
cover prompt
→ resampled hidden CoT h+ or h−
→ short, fixed cover answer
→ previously unavailable follow-up question
→ A/B answer
```

At each sentence boundary:

1. Sample native candidate continuations using only the cover prompt.
2. For each candidate, roll out the complete cover interaction.
3. Insert the follow-up question afterward.
4. Estimate its effect on the follow-up answer.
5. Retain candidates whose effect survives everything in between.

So the value function is approximately:

\[
V(h)=\mathbb{E}[\text{desired follow-up answer}\mid h,\ \text{cover completion},\ \text{follow-up}]
\]

This makes stability the optimization target rather than an accidental property.

I would use a small beam or particle population instead of keeping only the best sentence each round. Initially give every candidate a few rollouts, discard the clearly weak ones, and spend most of the evaluation budget distinguishing the finalists. Then evaluate the selected branches again using completely fresh samples to correct for optimizer’s curse.

## One essential implementation constraint

There must be an actual causal channel between the selected CoT and the follow-up.

A transformer does not retain a free-floating activation state across independent requests. The effect can survive if:

- The hidden CoT tokens remain in the model’s context.
- The same KV cache/sequence is continued.
- The API explicitly preserves reasoning state between the relevant items.

If the follow-up is a new request containing only an identical visible cover answer, while the hidden CoT has been discarded, the expected effect is zero. For the cleanest experiment, I would use an open-weight reasoning model and preserve the complete `<think>…</think>` sequence internally while hiding it from the simulated user.

## Stage the difficulty

| Condition | Intervening material | Purpose |
|---|---|---|
| 0 | Nothing | Replicate the original immediate effect |
| 1 | Fixed neutral text, 8–256 tokens | Measure raw decay with distance |
| 2 | Short canonical cover answer | First genuinely subliminal condition |
| 3 | Model-generated answer, increasing lengths | Test whether generation overwrites or transmits the bias |
| 4 | Tool/user boundary with context preserved | Test conversational persistence |
| 5 | Fresh context without hidden CoT | Negative control; should eliminate the effect |

This produces a persistence curve rather than one pass/fail number. Even a result like “large through 32 tokens, gone by 128 tokens” would be informative.

For the first positive-result attempt, I would use:

- One known 50/50 target question.
- A short, answer-only target response, so fresh target-side reasoning cannot wash out the effect.
- A cover task requiring substantial hidden reasoning but having a unique, one-token or fixed-format visible answer.
- Exactly the same visible answer in both arms.
- The same model and chat template throughout.

Then progressively allow a longer open-ended cover response and allow the target to reason.

## Make the target genuinely unavailable

A particularly clean setup is a tool boundary:

1. The model receives only a cover task.
2. Its resampled reasoning ends in an identical tool call.
3. The tool returns the 50/50 question.
4. The model answers it.

The model could not consciously plan around the target while generating its cover-task CoT because the target tokens were absent. The outer-loop search still uses the later answer as its selection reward. This is a strong form of subliminality, provided the tool continuation preserves the prior reasoning context.

## Exact targeting versus robust targeting

There are several increasingly strong claims:

1. **Exact-target carryover:** optimize for one exact follow-up and evaluate fresh rollouts of that question.
2. **Prompt robustness:** evaluate paraphrases and different response formats.
3. **Label robustness:** swap A/B positions while preserving the underlying alternatives.
4. **Behavioral generalization:** optimize on some questions and transfer to unseen questions.

The fourth will probably fail if the 50/50 questions are merely arbitrary independent preferences. There is no reason a state favouring tea over coffee should also determine window seat versus aisle seat.

A better approach is to find a shared behavioral axis:

- Collect responses to many binary questions under varied neutral contexts.
- Randomize label positions.
- Find clusters or principal components of correlated choices.
- Interpret promising factors—novelty versus familiarity, risk versus safety, simplicity versus feature richness, immediate versus delayed reward.
- Optimize the cover CoT using half the questions loading on a factor.
- Evaluate on held-out questions loading on the same factor.

That gives the resampling process a low-dimensional, naturally persistent model “mode” to target. It would be substantially more meaningful than a generic preference for the token `A`.

The recent steering-vector-distillation work is only an indirect, training-time prior here, but its result that traits well represented by a single direction transmit more readily supports choosing such a coherent behavioral factor rather than arbitrary independent choices. See [Subliminal Learning Is Steering Vector Distillation](https://arxiv.org/abs/2606.00995).

## Ways to increase the probability of seeing an effect

- Optimize across a distribution of bridge lengths and cover-answer variants. This selects branches that are robust rather than branches specialized to one exact token sequence.
- Start by selecting complete native CoTs from a large pool. Only use sentence-level refinement after identifying promising full traces.
- Stack multiple independent cover tasks resampled in the same direction. Test one, two, and four carriers. A cumulative effect would fit the broader observation that weak biases can influence many successive reasoning decisions rather than appearing in one decisive sentence. The hinted-question experiment in [Thought Branches](https://arxiv.org/html/2510.27484) is relevant evidence for that limited point.
- Keep the visible answer short initially. Long open-ended generation is both a stronger washout and a surface-text confound.
- Use direct label log-odds for scoring when available. If the target itself reasons before answering, use complete target rollouts instead.
- Generate several independently optimized positive/negative branch pairs. Thousands of probe samples from one selected pair are not independent experimental units.

## Important controls

The strongest version should include:

- **Identical surface output:** the visible cover answer is byte-for-byte identical across positive and negative arms.
- **Surface leakage test:** a classifier or blinded judge should be unable to predict the arm from the visible response.
- **Hidden-CoT deletion:** remove the selected hidden CoT while retaining the visible answer. The effect should disappear.
- **Random-branch control:** compare against ordinary on-policy CoTs matched for length, likelihood, and search budget.
- **Search/evaluation split:** never report the rollouts used to select the winning branches as evidence.
- **Label and option swapping:** distinguish semantic preference from first-option or `A` bias.
- **Paraphrase evaluation:** test whether the effect survives target wording changes.
- **Reset-context control:** confirms whether the phenomenon is in-context carryover rather than something persistent in the model weights.
- **Optimization-budget curve:** show how separation scales with samples spent.
- **Cover-answer-only selection:** select neutral visible responses without CoT resampling using the same reward. This establishes whether CoT is a special carrier or merely one form of prefix optimization.

The primary statistic should be the held-out positive-versus-negative log-odds difference, with uncertainty clustered by independently optimized branch pair and target question—not by individual rollout.

## Use the steering vector diagnostically

The earlier activation experiment could be valuable here without using the vector as the intervention. Measure projection onto the target direction:

```text
after selected CoT
after visible cover answer
after follow-up prompt
immediately before A/B
```

That would provide a mechanistic decay curve. It could reveal whether:

- The selected CoT creates the intended direction.
- The cover answer erases it.
- The follow-up reconstructs it.
- Behavioral persistence exists despite loss of the original vector.

## Other worthwhile variants

- **Canonicalization test:** replace the generated cover answer with a standardized paraphrase before asking the target. This removes much of the surface style channel.
- **Format-preserving domain shift:** optimize with one family of cover tasks and evaluate with unrelated cover tasks sharing only response format.
- **Repeated refresh:** insert several unrelated cover tasks, each independently resampled toward the same target, before the final probe.
- **Resampled-data fine-tuning:** train a same-base student on target-selected but semantically neutral cover traces. This becomes a bridge to conventional subliminal learning, although it is a different claim from inference-time persistence.
- **Arbitrary-bit capacity:** target several independent binary questions and measure how many bits can be transmitted through a fixed cover response. This is scientifically interesting, although less directly about stable behavioral steering.

## Recommended first implementation

Start with:

- Preserved hidden context.
- A surprise tool-injected target.
- A fixed one-token cover answer.
- Delayed-reward full-trace selection.
- A distance/bridge-length sweep.

Once that works, move to open-ended cover responses and held-out semantic-axis questions.

## Useful inputs from the earlier experiment

The most useful materials for refining and implementing this design would be:

- The exact model and chat template.
- How partial CoTs were resumed.
- Sampling and resampling budgets.
- The selected 50/50 questions.
- Fresh-sample effect sizes.
- Tests under label swaps or prompt paraphrases.
- The earlier implementation code and result artifacts.
