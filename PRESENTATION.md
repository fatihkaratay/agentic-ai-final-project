# Agentic Strategic Oversight of Classical Pathfinding

**5-minute class presentation — Introduction to Agentic AI, Module 14**
**Author:** Fatih Karatay

> Each slide includes a **Talking script** (≈ what to say aloud) and **On-screen** (what to display). Total target: ~5:00.

---

## Slide 1 — Title (≈ 15 sec)

### On-screen
- **Agentic Strategic Oversight of Classical Pathfinding**
- *Cross-episode episodic memory + LLM-driven reflective reasoning over A\**
- Fatih Karatay — Final Project, Module 14

### Talking script
> "My project asks a simple question: can a LangGraph agent make a classical pathfinder smarter, more adaptive, and — most importantly — *explainable*? I built a hybrid system where an LLM supervises an A\* planner across episodes, and benchmarked it against D\* Lite."

---

## Slide 2 — The problem (≈ 45 sec)

### On-screen
**A\* and D\* Lite are powerful, but limited:**
- **Stateless across episodes** — the corridor that nearly killed you yesterday is rediscovered from scratch tomorrow.
- **No human-readable rationale** — they return a path, not a *reason*.

**Research questions:**
1. **RQ1 (Memory):** Does cross-episode memory beat D\* Lite as episodes accumulate?
2. **RQ2 (Reflection):** Does LLM reflection beat raw memory writes?
3. **RQ3 (Cost-Quality):** When does LLM oversight justify its latency?

### Talking script
> "Classical replanners like D\* Lite are state-of-the-art, but they share two limitations. First, they reset between episodes — there's no learning across runs. Second, they produce paths, not explanations. As soon as you put a human in the loop, both gaps matter. I framed the project around three research questions about memory, reflection, and the cost-quality tradeoff."

---

## Slide 3 — System architecture (≈ 60 sec)

### On-screen
**LangGraph agent — 7 nodes sharing an `AgentState` TypedDict**

| Node | Type | Role |
|---|---|---|
| Environment Scanner | Reactive | Blends real-time obstacle proximity with persistent hazard map |
| A\* Planner | Deliberative | Plans/replans on cost-weighted grid |
| Memory Manager (read) | Memory | Retrieves up to 5 past-episode summaries |
| Path Risk Evaluator | **Deliberative-LLM** | Scores risk + writes natural-language justification |
| Path Healer | Deliberative | Bumps hazard on risky waypoints, triggers replan |
| Execution Monitor | Reactive | Steps forward, detects new obstacles |
| Reflection Node | **Reflective-LLM** | End-of-episode diagnosis, writes hazard cells to memory |

> *Show Figure 1 (LangGraph topology) here.*

### Talking script
> "The system is a hybrid deliberative-reactive agent built in LangGraph. Seven nodes share a typed state object. The two LLM-backed nodes are the heart of the design: the Path Risk Evaluator scores each planned path and writes a natural-language justification, and the Reflection Node runs at episode end to diagnose what happened and write hazard cells into a persistent memory store. Everything else — scanning, planning, healing, stepping — is deterministic. A\* still owns path geometry; the agent only adjusts cost weights and waypoint priorities."

---

## Slide 4 — How it learns (≈ 45 sec)

### On-screen
**Three mechanisms, three timescales:**

1. **Per-step uncertainty overlay** *(reactive)*
   Real-time + persistent hazard map → A\* cost weights.
2. **Bump-and-replan** *(deliberative, within episode)*
   LLM flags risky waypoints → Healer raises their cost → A\* replans.
3. **Reflection writes to persistent memory** *(reflective, between episodes)*
   `grid.hazard_map` survives `grid.reset()` — the *only* cross-episode learning signal.

### Talking script
> "Adaptation happens at three timescales. Reactively, the scanner overlays the persistent hazard map onto every plan. Deliberatively, the LLM can flag risky segments mid-episode, the Healer bumps their cost, and A\* replans within the modified cost space. Reflectively, at episode end the LLM identifies cells worth remembering and writes them to a hazard map that persists across episode resets — that's the cross-episode memory mechanism."

---

## Slide 5 — Experimental setup (≈ 30 sec)

### On-screen
- **4 variants × 3 conditions × 50 sequential episodes = 600 episodes**
- **Variants:** A\*, D\* Lite, Agent-NoReflection, **Full Agent**
- **Conditions:** Static (15%), Dynamic-Low (15%, slow), Dynamic-High (20%, every step)
- **Grid:** 20×20, identical seeds across variants
- **LLM:** `gpt-4o-mini`, ~2,500 calls, ~$1 USD total

### Talking script
> "Four variants, three difficulty tiers, fifty sequential episodes per cell — six hundred episodes in total. All variants use identical seed sequences for fair comparison. The whole experiment costs about a dollar in OpenAI tokens."

---

## Slide 6 — Results (≈ 60 sec)

### On-screen
**Success rate by variant and condition:**

| Variant | Static | Dynamic-Low | Dynamic-High |
|---|---|---|---|
| A\* | 96% | 20% | 0% |
| **D\* Lite** | 96% | **48%** | **6%** |
| Agent-NoReflection | 96% | 36% | 0% |
| **Full Agent** | 96% | 38% | 0% |

**Key findings:**
- D\* Lite wins on raw success rate.
- Reflection adds a consistent **+2 pp** over no-reflection.
- Learning curve: Full Agent climbs **0 → ~80%** rolling SR over first 15 episodes.
- Full Agent is **~1,500× slower** than D\* Lite per episode (18.9 s vs 13 ms).

> *Show Figure 4 (success rate) and Figure 5 (learning curve).*

### Talking script
> "Here's the honest result: D\* Lite wins on raw success rate. Forty-eight percent versus our thirty-eight on the dynamic-low tier. The reflection ablation gave a small but consistent two-percentage-point boost. But — and this is the key finding — the rolling success rate of the Full Agent climbs from zero to about eighty percent over the first fifteen episodes. That's direct empirical evidence that the cross-episode memory is doing real work. D\* Lite has no such trend, by design — it resets every episode."

---

## Slide 7 — The honest tradeoff & takeaways (≈ 60 sec)

### On-screen
**What we bought for the 1,500× slowdown:**
- Per-decision **natural-language risk justifications**
- End-of-episode **reflective diagnoses** stored to JSONL
- A complete **audit trail** for every episode — *no algorithmic baseline produces this*

> *Show Figure 7 (sample reflection summary).*

**Lessons learned:**
- Cross-episode memory works mechanically — visible in the learning curve.
- Bump-and-replan **over-corrects** in mild dynamics: longer paths → more exposure → more collisions.
- **Interpretability is the agent's defensible contribution**, not raw success rate.

**Future work:** graduated heal magnitudes · 50×50 grids · Agent-NoMemory ablation · semantic memory retrieval.

### Talking script
> "So why pay 1,500× the latency for worse success? Because the agent produces something no algorithm can: a complete, human-readable audit trail. Every routing decision has a rationale; every episode ends with a synthesized diagnosis. That's the agent's defensible contribution. The proposal anticipated this exact outcome — and I'd argue this *is* the agentic AI lesson: bounded rationality, cost-aware reasoning, and interpretability as a first-class output. Thank you — happy to take questions."

---

## Timing summary

| Slide | Topic | Time |
|---|---|---|
| 1 | Title | 0:15 |
| 2 | Problem & RQs | 0:45 |
| 3 | Architecture | 1:00 |
| 4 | How it learns | 0:45 |
| 5 | Experimental setup | 0:30 |
| 6 | Results | 1:00 |
| 7 | Tradeoff & conclusion | 1:00 |
| **Total** | | **~5:15** |

> Trim Slide 3 (drop one row of the table) or Slide 7 (skip the future work line) if running long.

---

## Anticipated Q&A

**Q: Why does D\* Lite beat your agent if memory is supposed to help?**
A: Two reasons. First, dynamic-low isn't hard enough — D\* Lite's incremental replanner handles it well, and the agent's bump-and-replan over-corrects, producing longer paths with more exposure. Second, our scope is 50 episodes; the memory benefit is most visible in the *learning curve* (Fig. 5), where the agent reaches ~80% rolling SR — D\* Lite shows no such trend.

**Q: Why use an LLM for risk assessment instead of a Bayesian heuristic?**
A: Three reasons. (1) The LLM synthesizes heterogeneous signals — hazard scores, path geometry, retrieved episode text — that a fixed heuristic can't generalize across. (2) Interpretability is a *primary* output, not a side effect. (3) LLM invocation is gated behind a risk threshold and operates on a compressed text summary, so cost stays bounded.

**Q: Is non-determinism in the LLM a problem?**
A: I treat it as bounded rationality, not a defect. Production deployments would need to characterize variance, but for a research proof-of-concept the variance is part of what makes this an *agentic* — rather than purely algorithmic — system.

**Q: What would you do differently?**
A: Tune the heal mechanism — currently it's binary (bump cost hard, replan). A graduated response proportional to LLM confidence should reduce over-correction. I'd also enable the conference scope: 50×50 grids, 200 i.i.d. episodes, and the Agent-NoMemory ablation to isolate the LLM's contribution from the persistent hazard map's contribution.
