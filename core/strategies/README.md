# core/strategies — B1–B4 Prompting Strategy Implementations

Each strategy is an `LLMJudgeBase` subclass with a unified `evaluate(nl, formula, ...) -> dict` API returning `{quality_score, reasoning, comment, refinement, logits, cost}`.

## Modules

| File                  | Purpose |
|-----------------------|---------|
| `base.py`             | `JudgeMethod` ABC and shared scaffolding |
| `strategy.py`         | B1–B4 implementations + B0 trivial baselines |
| `prompts/common.py`   | Shared prompt fragments (system instructions, output schema descriptions) |
| `prompts/few_shot_examples.py` | Curated few-shot examples (used by B2, B3) |
| `prompts/rendered/`   | Pre-rendered system prompts: `b{1-4}_v{1-4}_cfgB.txt` (Configuration B only) |

## Prompting Strategies

| ID | Class                  | Idea |
|----|------------------------|------|
| B0 | `B0_AlwaysYes` / `B0_AlwaysNo` / `B0_Random` / `B0_MajorityClass` | Trivial baselines (Pipeline 2 only) |
| B1 | `B1_ZeroShotCoT`       | Zero-shot, chain-of-thought reasoning |
| B2 | `B2_FewShot`           | Few-shot exemplars, no CoT |
| B3 | `B3_FewShotCoT`        | Few-shot + CoT |
| B4 | `B4_FeatureBased`      | Decompose into syntactic / ontological / truth-conditional / ambiguity sub-checks; aggregate |

## Configuration

Each strategy takes `configuration: str = "B"` (default = Task B = verdict + refinement) and `prompt_version: int ∈ {1,2,3,4}`.

| Variant | Style |
|---------|-------|
| v1      | Direct, lenient verdict threshold |
| v2      | Strict, epistemic humility (default to `?` when unsure) |
| v3      | Structured analysis with section headers, lenient |
| v4      | Adversarial (assume φ̂ is wrong, try to falsify), strict |

Configuration A (verdict only, no refinement) was an earlier experiment. Its rendered prompts are kept in `prompts/rendered/_Archive/` for reference but the strategy classes default to Configuration B.

## Logit-based decisions

When `logprobs=True`, the verdict `?` can be promoted to `yes`/`no` if the first answer-token logprob exceeds a threshold (default 0.7).
