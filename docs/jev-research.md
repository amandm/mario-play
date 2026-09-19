# Jev and our Mario learning project

Research date: 19 September 2026. This is a feasibility review; no Jev API calls,
training jobs, purchases, or integrations were performed for this review.

The subsequent [Jev-assisted PPO experiment](jev-assisted-results.md) reports
the implementation and measured learning results that followed this review.

## Finding

Jev can choose actions for a game agent. Its public interface does not currently
let us update its weights using our gameplay rewards. For our goal of observing
reinforcement learning, retain the existing PPO learner. Calling a pretrained
controller repeatedly is not evidence that it learns during play.

An **action space** is the set of allowed actions. An **action** is one selection
from that set. A **policy** maps an observation to action probabilities. **PPO**
updates the policy using rewards from collected gameplay.

## Evidence from public implementations

| Source | What exists | What the evidence establishes |
| --- | --- | --- |
| [TypeSafe Mario](https://github.com/fhshaik/typesafe-mario/tree/ca22449ed187118d19326d1f54b01b6636578aa4) | Emulator state becomes structured JSON; Jev chooses among seven controller actions. | A concrete inference integration. The inspected code has no policy training loop or evidence of improvement from experience. |
| [TypeSafe's launch demos](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | Doom control from structured state and Wikipedia link selection. | Vendor demonstrations of decision-making, not customer training. |
| [Parallel's hands-on account](https://x.com/everythingmeta/status/2101058921989390395) | Search reranking and classification experiments. | The author reports useful reranking results and uneven classification performance. These results do not establish Mario skill or learning. |

The Mario implementation computes motion and jump-timing facts in code, including
when a jump must begin. Consequently, observed behavior belongs to the whole
controller and its engineered state representation, not solely to Jev.

Related public posts included [Jev product ideas](https://x.com/gregisenberg/status/2101284640828915995),
an author's [ad-blocking demo](https://x.com/iam_zachi/status/2100529273186472318),
and a [Laya announcement](https://x.com/0xCVYH/status/2101171688683585622).
The product-ideas post is a proposal list, not evidence that those products work.

## Access, training, and cost

[TypeSafe's model reference](https://docs.typesafe.ai/models) documents Jev 1.13
as a hosted, text/JSON-input model. It explicitly excludes customer fine-tuning
and LoRA adaptation. The advertised RLCD training is TypeSafe's own training
process. Request instructions can change behavior without changing model weights.
The listed API price is $0.042 per million input tokens, with outputs free.

[The published customer agreement](https://typesafe.ai/legal/mca), section 2.3(b),
restricts distillation and training a model to imitate service outputs. Therefore,
Jev-generated expert-action labels are not an assumed available route for training
our policy. Section 8.2 describes discretionary promotional credits and opt-in
automatic refills. Verify available API allowance separately before running
experiments; Colab Pro compute allowance does not provide TypeSafe API credit.

## Fit with this repository

Our action mapping already has the same seven choices: no-op, right,
right+jump, right+run, right+run+jump, jump, and left. A future controller could
translate our original game's observations into a compact semantic description
and use Jev's selected action. No emulator or external game assets are needed.

For a small controller demonstration, use one fixed level, at most 100–200
decisions, one action-choice question per request, and pause the simulation while
waiting for responses. Pin the model version and record state, chosen action,
probabilities, and outcome. Only run after verifying sufficient free API credits
and that purchases cannot occur. This proposal is not a training experiment or
a claim of successful completion.

Do not insert externally chosen actions into ordinary PPO rollouts while retaining
the PPO policy's original action log-probabilities. That breaks the relationship
between the behavior policy and PPO's update. Our environment already supplies
measurable rewards, so adding a remote reward judge also needs a concrete benefit
before introducing complexity.

## Related open-weight model

[Laya](https://github.com/NandhaKishorM/laya) is a separate Apache-2.0 project,
with a 421-million-parameter English model and fine-tuning code. It offers a
more inspectable starting point if training a text-based decision model becomes
the objective. Its README acknowledges weak base-checkpoint performance on
typed decisions and overconfident probabilities. Its Jev comparisons use
different sources and workloads, so they do not establish superiority.

The supplied fine-tuning notebook targets two T4 GPUs and labeled decision tasks.
It is not a ready-made Mario PPO trainer. Single-GPU adaptation and gameplay
training would need engineering and validation; neither was attempted here.

For the present educational goal, use the existing PPO checkpoint and its
recorded behavior to explain observation, action, reward, and policy updates.
