# Final Project: Design and Demonstration of an Agentic AI System

## Purpose

This project serves as the culmination of the course, integrating all principles of Agentic AI, from foundational theory to practical system design. You will conceive, implement, and analyze a complete agentic system that demonstrates autonomy, reasoning, and adaptability. The system should align with the concepts taught throughout the course and reflect your understanding of how agency, rationality, learning, uncertainty, and reasoning interact within an intelligent system.

The focus is on the final system and its conceptual rigor, as in a senior design or research project. All theoretical and design considerations (PEAS, architecture, reasoning framework, etc.) should inform your implementation, but appear naturally through your Methods and Analysis sections rather than as checklists.

## Deliverble Format

Each student will submit a research-paper–style report (8–12 pages, IEEE or similar format). The written report should follow the structure of a scholarly paper:

### 1. Abstract (150–250 words)

A concise summary of your agentic system, its objective, core architecture, and primary outcomes.

### 2. Introduction

- Motivate the problem and explain its relevance.
- Situate your system within the broader context of agentic AI, why this problem benefits from an agentic approach.
- End with clear research goals or hypotheses (e.g., “This project explores how hybrid agent architectures with retrieval and reasoning can improve adaptability in [domain].”).

### 3. Background and Related Work

- Briefly discuss theoretical grounding: PEAS framework, agent architectures (reactive/deliberative/hybrid), rationality and decision making, and relevant literature or tools (e.g., LangGraph, CrewAI, HuggingFace Agents, RAG frameworks).
- Position your design choices within this context—how your agent aligns or diverges from established models.

### 4. System Design and Methods

- Describe your system architecture—the agents, their roles, and how they interact (e.g., orchestrator, planner, reasoner, validator, etc.).
- Explain the reasoning process (planning, reflection, adaptation, or reinforcement).
- Summarize your knowledge and uncertainty handling (retrieval, memory, probabilistic or rule-based inference).
- Discuss any learning components, such as online adaptation or reinforcement signals.
- Include diagrams or flowcharts where useful (e.g., LangGraph node structure, data flow, reasoning pipeline).
- Describe your experimental setup or environment, including datasets, evaluation contexts, or interaction loops.

### 5. Experiments and Evaluation

- Present the tasks or benchmarks your system performs.
- Define and justify the metrics you used to assess performance (quantitative and/or qualitative).
- Include any ablations or comparisons (e.g., with vs. without reasoning, or RAG on/off).
- Report results clearly through figures, tables, or logs.
- Discuss observed behaviors, emergent dynamics (if multiagent), and cost or latency patterns if relevant.

### 6. Results and Analysis

- Interpret your findings: What does the system reveal about agentic design?
- Highlight evidence of reasoning, adaptability, or autonomy.
- Reflect on trade-offs (e.g., reactivity vs. deliberation, bounded rationality, decision latency vs. quality).
- Discuss implications for real-world or domain-specific applications.

### 7. Conclusion and Future Work

- Summarize what you accomplished and what was learned.
- Propose potential extensions or generalizations.
- Reflect on the ethical, safety, or interpretability considerations of your system.

### 8. References

- Cite all frameworks, papers, or datasets used.
- Include citations for any external LLMs, APIs, or retrieval sources.

## Implementation and Expectation

- **Framework**: LangGraph preferred, but alternatives (CrewAI, HuggingFace Agents, custom orchestration) are acceptable.
- **Originality**: The agentic system must be your own design and implementation.
- **Modularity**: Your system should be composable, distinct reasoning or planning elements identifiable in the codebase.
- **Evaluation**: Demonstrate capability through a small but convincing proof of concept. The project’s sophistication lies in its reasoning and architecture, not dataset size or compute.
- **Documentation**: Provide a README with setup instructions and example runs.

## Submission

- **Paper (PDF)**: Submitted via Canvas.
- **Repository (GitHub or Zip)**: Containing code, configs, and evaluation logs.

## Evaluation Criteria (100 pts total)

| Category                       | Description                                                                 | Weight |
| ------------------------------ | --------------------------------------------------------------------------- | ------ |
| Concept & Motivation           | Clarity, originality, and relevance of the problem framed within agentic AI | 15     |
| System Design & Integration    | Quality of the architecture and coherence of agentic components             | 25     |
| Implementation & Functionality | Working demonstration, evidence of autonomy or reasoning                    | 20     |
| Evaluation & Analysis          | Depth of experimental analysis, metrics, and insights                       | 20     |
| Clarity & Presentation         | Organization, readability, figures, academic tone                           | 10     |
| Reflection & Ethics            | Awareness of system limitations, safety, and future implications            | 10     |

## Project Proposal: Interpretable Agentic Oversight of Classical Pathfinding with Cross-Episode Episodic Memory

### 1. Abstract

Classical dynamic replanning algorithms such as D\* Lite and LPA\* efficiently replan paths in response to environmental changes within a single episode, but they are stateless across episodes — each run begins without any knowledge of prior traversals. This project proposes a hybrid agentic system in which a LangGraph-based agent acts as a strategic supervisor over a classical A\* planner, augmenting it with two capabilities that algorithmic replanning cannot provide: (1) cross-episode episodic memory that builds a persistent hazard map from historical runs, and (2) reflective reasoning via a large language model (LLM) that produces an interpretable, natural-language audit trail for every trajectory decision. We evaluate the system through Monte Carlo simulations across three environment difficulty tiers (static, dynamic-low, dynamic-high) on 20×20 and 50×50 grids, comparing against standalone A\*, D\* Lite, and LPA\* baselines and ablating the memory and reflection components independently. We hypothesize that the full agent will outperform all algorithmic baselines in high-uncertainty, sequential-episode settings, while the natural-language justifications produced by the LLM reasoner provide a level of decision transparency unavailable from purely algorithmic approaches.

### 2. Novel Contribution Statement

This project makes three distinct contributions relative to prior work:

1. **Cross-Episode Episodic Memory for Path Planning**: Unlike D\* Lite and LPA\*, which reset all state at episode boundaries, the proposed system maintains a persistent hazard map updated by the Reflection Node after each episode. This enables the agent to encode environment-specific risk patterns over time — a form of online, non-parametric learning not present in classical replanning literature.

2. **LLM as Interpretable Strategic Reasoner**: Prior LLM-planning systems (LLM+P, SayPlan) use language models for high-level task decomposition or PDDL generation. This system deploys an LLM specifically for _trajectory-level risk assessment_, producing structured natural-language justifications that explain _why_ a path segment is flagged as risky. These justifications serve as a human-readable audit trail — addressing a concrete interpretability gap in autonomous navigation systems.

3. **Systematic Ablation of Agentic Components over Algorithmic Baselines**: By comparing six system variants (A\*, D\* Lite, LPA\*, Agent-NoMemory, Agent-NoReflection, Full Agent) under identical Monte Carlo conditions, this study provides the first systematic empirical breakdown of how much each agentic layer (reactive replanning, memory, reflective reasoning) contributes to performance over state-of-the-art classical baselines.

### 3. Research Questions

Rather than a single hypothesis, this project is structured around three focused research questions:

- **RQ1 (Memory)**: Does cross-episode episodic memory enable the agent to meaningfully outperform D\* Lite in high-uncertainty environments as the number of sequential episodes increases?
- **RQ2 (Reflection)**: Does the LLM Reflection Node's inter-episode analysis produce a measurable improvement in success rate over an agent with memory writes but no reflective synthesis?
- **RQ3 (Cost-Quality Trade-off)**: At what level of environment uncertainty does the quality gain from LLM-based strategic oversight justify its latency and token cost relative to pure algorithmic replanning?

### 4. Related Work Positioning

| System                                  | Approach                                                                  | Key Limitation vs. This Work                                             |
| --------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **A\*** (Hart et al., 1968)             | Optimal search in static graphs                                           | No dynamic replanning; no cross-episode learning                         |
| **D\* Lite** (Koenig & Likhachev, 2002) | Incremental replanning for dynamic environments                           | Stateless across episodes; no interpretable reasoning                    |
| **LPA\*** (Koenig et al., 2004)         | Lifelong incremental heuristic search                                     | Designed for a single continuously-changing map; no episodic memory      |
| **LLM+P** (Liu et al., 2023)            | LLM generates PDDL plans for classical solvers                            | Task-level planning only; no trajectory-level risk assessment            |
| **SayPlan** (Rana et al., 2023)         | LLM + scene graphs for 3D task planning                                   | High-level spatial reasoning; no cross-episode memory or grid navigation |
| **This Work**                           | LLM strategic oversight + episodic memory + reflective reasoning over A\* | —                                                                        |

The proposed system occupies a distinct niche: it targets _trajectory-level_ risk assessment (not task-level planning), operates in _episodic sequential settings_ (not single-run), and produces _interpretable decision justifications_ (not just optimized paths).

---

### 5. System Design: PEAS Framework

The PEAS framework defines the agent's operating context and rational objectives.

#### Performance Measure

- **Path Success Rate**: Percentage of runs where the agent reaches the goal without collision.
- **Path Efficiency Ratio**: Agent path length divided by the optimal A\* path length on an equivalent static grid (lower overhead = better).
- **Collision / Near-Miss Rate**: Frequency of entering cells that become blocked during traversal.
- **Replanning Count**: Number of times the agent invokes A\* mid-episode; reflects adaptability cost.
- **Decision Latency**: Average wall-clock time per agentic reasoning cycle (important for real-world feasibility).
- **Memory Utilization Score**: How often retrieved historical patterns led to a successful avoidance decision.
- **Decision Justification Quality**: Human-rated coherence and accuracy of LLM-generated path justifications (qualitative, sampled from 20 episodes per condition).

#### Environment

- **Grid World**: A discrete NxN grid (planned: 20×20 and 50×50 variants) where each cell is either free, statically blocked, or dynamically occupied.
- **Dynamic Obstacles**: Obstacles that move stochastically each time step according to configurable motion models (random walk, directional drift, periodic patterns). Obstacle density and mobility are varied across experimental conditions.
- **Uncertainty Layer**: Each cell carries a hazard probability score derived from historical obstacle visitation frequency, creating a probabilistic map the agent can consult.
- **Episodic Structure**: Each simulation run is one episode — the agent starts at a fixed origin, must reach a fixed goal, and the environment resets between episodes with new obstacle seeds drawn from the same distribution.

#### Actuators

- **Path Override Commands**: Instruct the execution layer to follow a modified waypoint sequence instead of the current A\* path.
- **Replanning Trigger**: Signal A\* to recompute from the current position with updated cost weights.
- **Memory Write**: Persist observed hazard events (cell coordinates, time step, obstacle type) to the episodic memory store.
- **Cost Map Adjustment**: Modify per-cell traversal costs fed into A\* to encode soft avoidance of historically risky zones.

#### Sensors

- **Grid State Snapshot**: Full or partial observability of current obstacle positions (configurable: full vs. limited radius).
- **Planned Path**: The current A\*-computed route as an ordered list of waypoints.
- **Uncertainty / Hazard Map**: A probabilistic heatmap of cell risk scores built from memory.
- **Step Counter & Episode History**: Current time step and record of past decisions within the episode, enabling within-episode reflection.
- **Memory Retrieval Output**: Relevant past episodes retrieved by the Memory Manager node, providing context for pattern recognition.

#### Why an LLM — Not Another Algorithm

A critical design question is why an LLM is used for risk assessment rather than a purely algorithmic approach (e.g., a Bayesian occupancy grid or potential field). The justification is threefold:

1. **Semantic Pattern Recognition**: The LLM can synthesize multiple heterogeneous signals (hazard map scores, path geometry, retrieved episode summaries) into a coherent risk narrative that crosses arbitrary obstacle configurations — something a fixed heuristic cannot generalize across.
2. **Interpretability as a First-Class Output**: The LLM's natural-language justification is not a side effect — it is a primary contribution. No algorithmic replanner produces human-readable explanations for _why_ a specific path segment is risky.
3. **Bounded LLM Invocation**: The LLM is gated behind a risk threshold and operates on a compressed grid summary (not raw matrix data), keeping latency and token costs tractable. This is a design choice — not an assumption that LLMs are always the right tool for spatial reasoning.

---

### 6. Proposed Architecture: LangGraph Workflow

The system is a hybrid deliberative-reactive agent implemented as a directed graph of LangGraph nodes. A\* provides fast, locally optimal path computation; the LangGraph orchestrator provides strategic oversight, memory, and reflective re-evaluation.

#### Node Descriptions

| Node                    | Role                                                                                                                                                                                                                               | Type               |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| **Environment Scanner** | Reads the current grid state; computes a per-cell uncertainty score by blending real-time observation with the historical hazard map. Emits a structured `EnvironmentState` object.                                                | Reactive           |
| **A\* Planner**         | Runs A\* search on the current grid using the uncertainty-adjusted cost map. Returns an ordered path. Can be called at episode start or triggered mid-episode for replanning.                                                      | Deliberative       |
| **Path Risk Evaluator** | LLM-backed node that inspects the planned path against the uncertainty map and retrieved memory. Outputs a risk score (0–1) and a natural-language justification. Acts as the strategic reasoner.                                  | Deliberative / LLM |
| **Memory Manager**      | Manages an episodic memory store (e.g., a vector store or structured log). On read: retrieves historically similar grid configurations and their outcomes. On write: records the current episode's hazard events after completion. | Memory             |
| **Path Healer**         | Invoked when risk exceeds a configurable threshold. Proposes waypoint-level modifications (detours, holds) or requests a full replan from A\*. Validates that the healed path is still goal-reachable.                             | Deliberative       |
| **Execution Monitor**   | Steps the agent along the active path one cell at a time. Detects newly revealed obstacles (sensor updates mid-path). Conditionally re-routes control back to the Path Risk Evaluator.                                             | Reactive           |
| **Reflection Node**     | Runs at episode end. Compares predicted risks to actual outcomes, identifies memory gaps, and writes a structured episode summary to the Memory Manager. Enables inter-episode learning.                                           | Reflective         |

#### Graph Topology (Conceptual Flow)

```
[Environment Scanner]
        |
        v
  [A* Planner] <-------------------+
        |                          | (replan trigger)
        v                          |
[Memory Manager (read)]            |
        |                          |
        v                          |
[Path Risk Evaluator]              |
        |                          |
   risk < threshold?               |
     /         \                   |
   YES           NO                |
    |             |                |
    |       [Path Healer] ---------+
    |
    v
[Execution Monitor]
        |
   obstacle detected?
     /         \
   YES           NO
    |             |
[Risk Evaluator]  v
(mid-episode)  [Goal Reached]
                  |
                  v
          [Reflection Node]
                  |
          [Memory Manager (write)]
```

#### State Schema

The shared LangGraph `AgentState` object carries:

- `grid`: current NxN obstacle matrix
- `current_pos`: agent's live coordinates
- `planned_path`: list of waypoints from A\*
- `hazard_map`: NxN float array of cell risk scores
- `risk_score`: current evaluator output
- `risk_justification`: LLM-generated natural-language explanation for the risk score
- `memory_context`: retrieved past episodes
- `episode_log`: timestamped record of decisions and outcomes
- `replan_count`: integer counter for cost tracking

#### Key Design Decisions

- **Bounded Rationality**: The LLM evaluator operates on a compressed summary of the path and hazard map (not the raw grid matrix) to keep token costs manageable.
- **Separation of Concerns**: A\* remains the sole path-geometry engine; the agent never overrides geometry, only cost weights and waypoint priorities.
- **Conditional Edges**: LangGraph conditional edges route the graph based on `risk_score` and `obstacle_detected` flags, keeping reactive and deliberative loops cleanly separated.
- **Memory as Soft Priors**: Historical patterns inform A\* cost weights rather than hard constraints, preserving A\*'s optimality guarantees within the adjusted cost space.
- **Justification Logging**: Every LLM risk evaluation stores its natural-language justification in `episode_log`, creating a full audit trail for post-hoc interpretability analysis.

---

### 7. Experimental Methodology

#### Environment Configurations

Three environment difficulty tiers will be evaluated across both grid sizes (20×20, 50×50):

| Tier             | Obstacle Density | Obstacle Mobility                                  | Description                                               |
| ---------------- | ---------------- | -------------------------------------------------- | --------------------------------------------------------- |
| **Static**       | 15%              | None                                               | Baseline; A\* should achieve near-perfect performance.    |
| **Dynamic-Low**  | 15%              | Slow random walk (1 cell/5 steps)                  | Mild uncertainty; tests basic reactive replanning.        |
| **Dynamic-High** | 25%              | Fast random walk (1 cell/step) + periodic spawning | High uncertainty; tests memory and reflective adaptation. |

#### Simulation Protocol

- **Monte Carlo Runs**: 200 independent episodes per condition (600 total per system variant), with different random seeds for obstacle initialization but identical seeds across variants for fair comparison.
- **Sequential Episode Batches**: For RQ1 and RQ2, episodes are run sequentially (not shuffled) so that cross-episode memory accumulates naturally across runs. This is distinct from the i.i.d. Monte Carlo protocol and is analyzed separately.
- **Observability**: Experiments run under both full observability and limited-radius (5-cell) observability to test sensor constraints.
- **Episode Termination**: Success (goal reached), Failure-Collision (agent enters blocked cell), or Timeout (steps exceed 3× optimal path length).

#### Baselines and Ablation Study

Six system variants will be compared to isolate individual contributions:

| Variant                | Components Active                                                                | Purpose                                                                                |
| ---------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Baseline A\***       | A\* Planner + Execution Monitor (no replanning on obstacle detection)            | Theoretical lower bound in dynamic environments                                        |
| **D\* Lite**           | Incremental dynamic replanning (Koenig & Likhachev, 2002)                        | Primary algorithmic competitor; state-of-the-art for dynamic single-episode navigation |
| **LPA\***              | Lifelong Planning A\* (Koenig et al., 2004)                                      | Secondary algorithmic competitor; tests if incremental heuristic search closes the gap |
| **Agent-NoMemory**     | All LangGraph nodes active, Memory Manager disabled                              | Isolates the value of cross-episode memory                                             |
| **Agent-NoReflection** | All LangGraph nodes active, Reflection Node disabled (writes only, no synthesis) | Isolates the value of reflective reasoning over raw memory                             |
| **Full Agent**         | All nodes active including Memory Manager and Reflection Node                    | Primary proposed system                                                                |

#### Evaluation Protocol

- All variants run on identical episode seeds for fair comparison.
- The LLM used for the Path Risk Evaluator and Reflection Node will be fixed (`claude-haiku-4-5`) across all agent variants.
- API call counts and token usage will be logged per episode to quantify computational overhead.
- For the interpretability metric, 20 episodes per condition (Dynamic-High only) will be sampled and justifications rated by two independent human raters on a 3-point coherence/accuracy scale (inter-rater agreement measured via Cohen's κ).

---

### 8. Key Metrics

| Metric                                   | Definition                                                                  | Goal                                                              |
| ---------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Success Rate (SR)**                    | Episodes reaching goal without collision / total episodes                   | Maximize; primary metric                                          |
| **Path Efficiency Ratio (PER)**          | Agent steps taken / A\* optimal steps on equivalent static grid             | Minimize overhead; target < 1.3                                   |
| **Collision Rate (CR)**                  | Episodes ending in collision / total episodes                               | Minimize                                                          |
| **Mean Replanning Count (MRC)**          | Average number of mid-episode A\* calls                                     | Monitor; high MRC may indicate reactive thrashing                 |
| **Decision Latency (DL)**                | Mean wall-clock time per agentic reasoning cycle (ms)                       | Characterize trade-off vs. quality                                |
| **Memory Hit Rate (MHR)**                | Fraction of risk evaluations where retrieved memory influenced the decision | Track memory utility over episodes                                |
| **Episode-over-Episode Learning Curve**  | SR rolling average across sequential episodes (batches of 10)               | Demonstrates inter-episode improvement from Reflection Node (RQ2) |
| **Token Cost per Episode**               | Total LLM tokens consumed / episode                                         | Quantify agentic overhead for RQ3                                 |
| **Decision Justification Quality (DJQ)** | Human-rated coherence and accuracy of LLM justifications (1–3 scale)        | Demonstrate interpretability contribution; target avg ≥ 2.5       |

Statistical significance will be assessed using two-proportion z-tests (for SR/CR) and Welch's t-tests (for continuous metrics) at α = 0.05 across the 200-episode samples.

---

### 9. Implementation Plan

#### Phase 1 — Grid Environment, A\*, D\* Lite, and LPA\* Baselines

- Implement the NxN grid world with configurable obstacle density and stochastic motion models.
- Implement standard A\* with a pluggable cost function (to later accept the hazard map overlay).
- Implement D\* Lite with full incremental replanning on obstacle detection.
- Implement LPA\* for comparison.
- Build the `Execution Monitor` loop with collision detection and timeout logic, shared across all variants.
- Validate all three baselines on static and dynamic environments before introducing the agent.
- **Deliverable**: Three working baselines (A\*, D\* Lite, LPA\*) with deterministic and stochastic environment modes and consistent logging.

#### Phase 2 — LangGraph Agent Scaffold

- Set up the LangGraph `StateGraph` with the `AgentState` schema.
- Implement the `Environment Scanner`, `A\* Planner`, and `Path Risk Evaluator` nodes (LLM integration via Anthropic API).
- Wire conditional edges for the risk threshold decision.
- Implement the `Path Healer` with waypoint detour logic.
- Log `risk_justification` strings to `episode_log` for interpretability analysis.
- **Deliverable**: Agent-NoMemory variant functional on Dynamic-Low environments.

#### Phase 3 — Memory Integration

- Implement the `Memory Manager` with an episodic log (JSON-backed store initially; optionally upgrade to a vector store for semantic retrieval).
- Connect memory reads to the `Path Risk Evaluator` context window.
- Connect memory writes from the `Reflection Node` post-episode.
- **Deliverable**: Agent-NoReflection variant functional; memory read/write verified across sequential episodes.

#### Phase 4 — Reflection & Adaptation

- Implement the `Reflection Node` with a structured prompt for episode analysis (outcome vs. prediction, pattern identification, memory gap annotation).
- Verify the Episode-over-Episode Learning Curve metric shows measurable improvement over 50+ sequential episodes.
- **Deliverable**: Full Agent variant functional.

#### Phase 5 — Experiments & Evaluation

- Run all six variants across all environment configurations and both grid sizes (full Monte Carlo + sequential episode protocol).
- Collect and aggregate metrics; generate plots for success rate curves, PER distributions, learning curves, and cost-quality scatter plots.
- Run ablation comparisons; apply statistical significance tests.
- Conduct human evaluation of 20 sampled justifications per condition (Dynamic-High).
- **Deliverable**: Complete results tables and figures ready for the paper.

#### Phase 6 — Paper & Repository Finalization

- Write the full IEEE-format paper following the deliverable structure above.
- Finalize the repository with a README, example run scripts, configuration files, and evaluation logs.
- **Deliverable**: Submitted PDF and GitHub repository.

#### Technology Stack

| Component              | Technology                                                     |
| ---------------------- | -------------------------------------------------------------- |
| Agent Orchestration    | LangGraph (Python)                                             |
| LLM Backend            | Anthropic API (`claude-haiku-4-5` for evaluator/reflector)     |
| Grid Environment       | NumPy + custom Python simulation                               |
| A\* / D\* Lite / LPA\* | Custom Python implementations                                  |
| Memory Store           | JSON log (Phase 3); optionally ChromaDB for semantic retrieval |
| Experiment Runner      | Python scripts with seed-controlled random                     |
| Visualization          | Matplotlib / Seaborn                                           |
| Statistical Analysis   | SciPy                                                          |

---

### 10. Conference Target & Submission Strategy

| Venue                                     | Track                          | Fit         | Notes                                                              |
| ----------------------------------------- | ------------------------------ | ----------- | ------------------------------------------------------------------ |
| **ICAPS Workshop on Planning & Learning** | Workshop short paper (4–6 pp.) | High        | Direct domain fit; welcomes hybrid neuro-symbolic systems          |
| **AAAI 2027 Student Abstract**            | 2-page abstract                | High        | Appropriate scope for a proof-of-concept contribution              |
| **FLAIRS 2026/2027**                      | Full paper (8 pp.)             | Medium-High | Applied AI venue; accepts solid empirical work with good ablations |
| **ECAI 2026 Workshop**                    | Workshop paper                 | Medium      | European AI; broader scope, good for novel system papers           |

**Recommended strategy**: Target the ICAPS Planning & Learning workshop as a short paper first. The D\* Lite comparison, three research questions, and interpretability metric make the scope appropriate for a 4–6 page workshop contribution. If results are strong, expand to a full paper for FLAIRS or ECAI.

---

### 11. Additional Considerations

#### Ethical and Safety Reflections

- The system's LLM component introduces **non-determinism**: the same grid state may yield different risk assessments across runs, which is worth analyzing as a form of bounded rationality rather than a flaw.
- **Interpretability**: The natural-language justifications produced by the Path Risk Evaluator and Reflection Node are a core artifact — they provide a human-readable audit trail of agent decisions, addressing the "black box" criticism of LLM-based agents.
- **Scope Limitation**: This system is a simulation study. Before applying similar agentic oversight to real-world robotics or autonomous vehicles, significantly more rigorous safety validation would be required.
- **Cost Awareness**: LLM API calls carry financial and latency costs. The design intentionally gates LLM invocation behind risk thresholds and uses a lightweight model (`haiku`) to make the overhead tractable at simulation scale.

#### Anticipated Challenges

- Prompt engineering for the Path Risk Evaluator to produce consistent, structured risk scores across diverse grid configurations.
- Balancing memory retrieval relevance vs. noise — overly broad memory retrieval may introduce false priors.
- Ensuring the reflection loop does not overfit to recent episodes at the expense of generalizable patterns.
- D\* Lite may outperform the Full Agent on per-episode metrics in Dynamic-Low conditions — this is an expected and honest result that the paper will frame as a cost-quality trade-off rather than a failure.

---

### 12. Class Project Alignment

This section maps the proposal directly to the course rubric to ensure the class submission is complete and well-targeted before pursuing any conference extension.

#### Rubric Coverage

| Category                           | Weight | How This Project Covers It                                                                                                                                                                                                                                         | Risk                              |
| ---------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------- |
| **Concept & Motivation**           | 15     | Clear problem (A\* fails in dynamic environments), novel agentic angle (cross-episode memory + interpretability), situated within agentic AI via PEAS + hybrid deliberative/reactive architecture, three focused RQs                                               | Low                               |
| **System Design & Integration**    | 25     | 7-node LangGraph graph with distinct roles (reactive Scanner, deliberative Planner/Healer, LLM Evaluator, Memory Manager, Reflective Node); conditional edges; shared `AgentState`; memory as soft priors; modularity enforced by one-file-per-node repo structure | Low                               |
| **Implementation & Functionality** | 20     | Working LangGraph agent demonstrating autonomy (self-directed replanning loop), reasoning (LLM risk evaluation + NL justifications), and adaptability (cross-episode memory). Class MVP scope (see below) is realistic.                                            | Medium — scope must be managed    |
| **Evaluation & Analysis**          | 20     | 9 defined metrics, 4-variant ablation study, D\* Lite comparison, Monte Carlo protocol, statistical significance tests, episode-over-episode learning curve, qualitative justification analysis                                                                    | Low                               |
| **Clarity & Presentation**         | 10     | IEEE format, 8-section structure, 7 planned figures, tables for all comparisons; PEAS woven naturally into narrative (not as checklist) per instructor guidance                                                                                                    | Medium — figures must be produced |
| **Reflection & Ethics**            | 10     | Non-determinism as bounded rationality, interpretability as core output, scope limitations (simulation only), cost awareness, anticipated challenges including honest D\* Lite failure case                                                                        | Low                               |

---

#### Class MVP Scope

The full experimental protocol is designed for conference submission. For the **class deadline**, the following reduced scope is sufficient to earn full marks while keeping implementation realistic:

| Parameter                  | Class MVP                                         | Conference Extension             |
| -------------------------- | ------------------------------------------------- | -------------------------------- |
| Grid sizes                 | 20×20 only                                        | + 50×50                          |
| Algorithmic baselines      | A\* + D\* Lite                                    | + LPA\*                          |
| Agent variants             | 4 (A\*, D\* Lite, Agent-NoReflection, Full Agent) | + Agent-NoMemory                 |
| Episodes per condition     | 50 (sequential)                                   | 200 (i.i.d. Monte Carlo)         |
| Difficulty tiers           | All 3 (Static, Dynamic-Low, Dynamic-High)         | Same                             |
| Observability              | Full only                                         | + Limited radius                 |
| Human justification rating | 5 sampled episodes (qualitative discussion)       | 20 episodes, 2 raters, Cohen's κ |
| Statistical tests          | Descriptive + basic t-test                        | Full z-test/t-test battery       |

The instructor's rubric states _"the project's sophistication lies in its reasoning and architecture, not dataset size or compute."_ The MVP above fully demonstrates autonomy, reasoning, adaptability, and agentic components — which is exactly what the rubric measures.

---

#### Paper Section Mapping

How proposal content maps to the required 8-section paper structure:

| Paper Section                    | Word Target   | Content Source                                                                                                                                                                                             |
| -------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Abstract**                  | 150–250 words | Condense Section 1 of this proposal; must mention D\* Lite comparison, cross-episode memory, and interpretability in one paragraph                                                                         |
| **2. Introduction**              | ~500 words    | Motivate A\*'s limitation in dynamic environments → why agentic approach → state RQ1, RQ2, RQ3 from Section 3; end with paper roadmap sentence                                                             |
| **3. Background & Related Work** | ~600 words    | Weave Section 4 (Related Work table) into prose; introduce PEAS naturally as the framework used to analyze the environment — not as a checklist; cite Hart 1968, Koenig 2002/2004, Liu 2023, Rana 2023     |
| **4. System Design & Methods**   | ~1000 words   | Section 6 (Architecture) as the primary source; embed PEAS components (sensors, actuators, environment) naturally within the architecture description; include LangGraph node diagram and data-flow figure |
| **5. Experiments & Evaluation**  | ~700 words    | Sections 7 + 8: environment configs table, variant comparison table, metrics table; reference figures by number                                                                                            |
| **6. Results & Analysis**        | ~800 words    | Populated after experiments; structured around RQ1, RQ2, RQ3; include learning curve plot, bar charts, sample LLM justification                                                                            |
| **7. Conclusion & Future Work**  | ~400 words    | Summarize what each RQ answered; propose extensions (larger grids, real-world robotics, RAG-based memory); reflect on non-determinism, interpretability, and safety from Section 11                        |
| **8. References**                | —             | Hart 1968 (A\*), Koenig & Likhachev 2002 (D\* Lite), Koenig 2004 (LPA\*), Liu et al. 2023 (LLM+P), Rana et al. 2023 (SayPlan), LangGraph docs, Anthropic API                                               |

> **Instructor note on PEAS**: The course states PEAS should "appear naturally through Methods and Analysis sections rather than as checklists." In the paper, do not have a section titled "PEAS Framework." Instead, introduce the performance measures in Section 5 (Evaluation), the environment in Section 4 (System Design), and sensors/actuators as part of the agent architecture description.

---

#### Required Figures

The rubric awards points for "figures, tables, academic tone." All seven figures below must appear in the final paper:

| #   | Figure                                | Type                                                          | Paper Section |
| --- | ------------------------------------- | ------------------------------------------------------------- | ------------- |
| 1   | LangGraph node architecture           | Proper directed graph (not ASCII — use `graphviz` or draw.io) | §4            |
| 2   | Agent reasoning pipeline / data flow  | Flowchart with `AgentState` passing between nodes             | §4            |
| 3   | Grid world visualization              | Rendered grid with obstacle heatmap overlay and planned path  | §4 or §5      |
| 4   | Success rate by variant and condition | Grouped bar chart (3 tiers × 4 variants)                      | §5/§6         |
| 5   | Episode-over-episode learning curve   | Line plot, rolling SR over sequential episodes                | §5/§6         |
| 6   | Cost-quality scatter                  | Decision latency vs. success rate per variant                 | §6            |
| 7   | Sample LLM justification              | Formatted text box showing a real Path Risk Evaluator output  | §6            |

---

#### Repository Structure

The rubric requires a documented repository with setup instructions, modularity, and evaluation logs. The following structure satisfies all three:

```
agentic-ai-final-project/
├── README.md                    # Setup, requirements, example runs (required by rubric)
├── requirements.txt
├── src/
│   ├── environment/             # Grid world, obstacle motion models
│   ├── planners/                # a_star.py, d_star_lite.py, lpa_star.py
│   ├── agent/                   # LangGraph agent
│   │   ├── nodes/               # One file per node — satisfies modularity requirement
│   │   │   ├── scanner.py
│   │   │   ├── planner.py
│   │   │   ├── risk_evaluator.py
│   │   │   ├── memory_manager.py
│   │   │   ├── path_healer.py
│   │   │   ├── execution_monitor.py
│   │   │   └── reflection.py
│   │   ├── state.py             # AgentState schema
│   │   └── graph.py             # StateGraph wiring and conditional edges
│   └── memory/                  # Episodic store read/write utilities
├── experiments/
│   ├── run_experiments.py       # Seed-controlled runner for all variants
│   └── configs/                 # Environment and agent configs (YAML)
├── results/
│   ├── logs/                    # Episode-level JSON logs (required by rubric)
│   └── figures/                 # Generated plots
└── paper/
    └── paper.pdf
```

The one-file-per-node structure directly demonstrates the **modularity** requirement: _"distinct reasoning or planning elements identifiable in the codebase."_
