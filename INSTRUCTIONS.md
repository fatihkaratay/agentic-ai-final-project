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

## Project Proposal: Agentic Strategic Oversight vs. Classical A\* Pathfinding

### 1. Abstract

This project investigates the integration of Agentic AI with classical pathfinding algorithms to improve navigation in dynamic, uncertain environments. While the A* algorithm provides mathematically optimal paths in static grids, it lacks the Memory and Reasoning capabilities required to anticipate environmental shifts. This study proposes a hybrid system where a LangGraph-based Agent acts as a strategic supervisor, utilizing the PEAS framework to evaluate and "heal" A* trajectories based on historical patterns and real-time uncertainty. We hypothesize that this agentic approach will achieve higher success rates in Monte Carlo simulations involving dynamic obstacles compared to standalone A\*.

### 2. Research Hypothesis

"An Agentic System utilizing reflective reasoning and memory will outperform the classical A\* algorithm in environments with high uncertainty and dynamic obstacles by optimizing for long-term safety over immediate path distance."

---

### 3. System Design: PEAS Framework

The PEAS framework defines the agent's operating context and rational objectives.

#### Performance Measure

- **Path Success Rate**: Percentage of runs where the agent reaches the goal without collision.
- **Path Efficiency Ratio**: Agent path length divided by the optimal A\* path length on an equivalent static grid (lower overhead = better).
- **Collision / Near-Miss Rate**: Frequency of entering cells that become blocked during traversal.
- **Replanning Count**: Number of times the agent invokes A\* mid-episode; reflects adaptability cost.
- **Decision Latency**: Average wall-clock time per agentic reasoning cycle (important for real-world feasibility).
- **Memory Utilization Score**: How often retrieved historical patterns led to a successful avoidance decision.

#### Environment

- **Grid World**: A discrete NxN grid (planned: 20×20 and 50×50 variants) where each cell is either free, statically blocked, or dynamically occupied.
- **Dynamic Obstacles**: Obstacles that move stochastically each time step according to configurable motion models (random walk, directional drift, periodic patterns). Obstacle density and mobility are varied across experimental conditions.
- **Uncertainty Layer**: Each cell carries a hazard probability score derived from historical obstacle visitation frequency, creating a probabilistic map the agent can consult.
- **Episodic Structure**: Each simulation run is one episode — the agent starts at a fixed origin, must reach a fixed goal, and the environment resets between episodes with new obstacle seeds.

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

---

### 4. Proposed Architecture: LangGraph Workflow

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
- `memory_context`: retrieved past episodes
- `episode_log`: timestamped record of decisions and outcomes
- `replan_count`: integer counter for cost tracking

#### Key Design Decisions

- **Bounded Rationality**: The LLM evaluator operates on a compressed summary of the path and hazard map (not the raw grid matrix) to keep token costs manageable.
- **Separation of Concerns**: A\* remains the sole path-geometry engine; the agent never overrides geometry, only cost weights and waypoint priorities.
- **Conditional Edges**: LangGraph conditional edges route the graph based on `risk_score` and `obstacle_detected` flags, keeping reactive and deliberative loops cleanly separated.
- **Memory as Soft Priors**: Historical patterns inform A\* cost weights rather than hard constraints, preserving A\*'s optimality guarantees within the adjusted cost space.

---

### 5. Experimental Methodology

#### Environment Configurations

Three environment difficulty tiers will be evaluated across both grid sizes (20×20, 50×50):

| Tier             | Obstacle Density | Obstacle Mobility                                  | Description                                               |
| ---------------- | ---------------- | -------------------------------------------------- | --------------------------------------------------------- |
| **Static**       | 15%              | None                                               | Baseline; A\* should achieve near-perfect performance.    |
| **Dynamic-Low**  | 15%              | Slow random walk (1 cell/5 steps)                  | Mild uncertainty; tests basic reactive replanning.        |
| **Dynamic-High** | 25%              | Fast random walk (1 cell/step) + periodic spawning | High uncertainty; tests memory and reflective adaptation. |

#### Simulation Protocol

- **Monte Carlo Runs**: 200 independent episodes per condition (600 total per system variant), with different random seeds for obstacle initialization.
- **Observability**: Experiments run under both full observability and limited-radius (5-cell) observability to test sensor constraints.
- **Episode Termination**: Success (goal reached), Failure-Collision (agent enters blocked cell), or Timeout (steps exceed 3× optimal path length).

#### Ablation Study (System Variants)

To isolate the contribution of each agentic component, four variants will be compared:

| Variant                | Components Active                                                            |
| ---------------------- | ---------------------------------------------------------------------------- |
| **Baseline A\***       | A\* Planner + Execution Monitor only (no agent reasoning)                    |
| **Agent-NoMemory**     | All nodes active, Memory Manager disabled                                    |
| **Agent-NoReflection** | All nodes active, Reflection Node disabled (memory writes only, no analysis) |
| **Full Agent**         | All nodes active including Memory Manager and Reflection Node                |

#### Evaluation Protocol

- All variants run on identical episode seeds for fair comparison.
- The LLM used for the Path Risk Evaluator and Reflection Node will be fixed (e.g., `claude-haiku-4-5` for cost efficiency) across all agent variants.
- API call counts and token usage will be logged per episode to quantify computational overhead.

---

### 6. Key Metrics

| Metric                                  | Definition                                                                  | Goal                                                        |
| --------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **Success Rate (SR)**                   | Episodes reaching goal without collision / total episodes                   | Maximize; primary metric                                    |
| **Path Efficiency Ratio (PER)**         | Agent steps taken / A\* optimal steps on equivalent static grid             | Minimize overhead; target < 1.3                             |
| **Collision Rate (CR)**                 | Episodes ending in collision / total episodes                               | Minimize                                                    |
| **Mean Replanning Count (MRC)**         | Average number of mid-episode A\* calls                                     | Monitor; high MRC may indicate reactive thrashing           |
| **Decision Latency (DL)**               | Mean wall-clock time per agentic reasoning cycle (ms)                       | Characterize trade-off vs. quality                          |
| **Memory Hit Rate (MHR)**               | Fraction of risk evaluations where retrieved memory influenced the decision | Track memory utility over episodes                          |
| **Episode-over-Episode Learning Curve** | SR rolling average across sequential episodes                               | Demonstrates inter-episode improvement from Reflection Node |
| **Token Cost per Episode**              | Total LLM tokens consumed / episode                                         | Quantify agentic overhead for cost analysis                 |

Statistical significance will be assessed using two-proportion z-tests (for SR/CR) and Welch's t-tests (for continuous metrics) at α = 0.05 across the 200-episode samples.

---

### 7. Implementation Plan

#### Phase 1 — Grid Environment & A\* Baseline

- Implement the NxN grid world with configurable obstacle density and stochastic motion models.
- Implement standard A\* with a pluggable cost function (to later accept the hazard map overlay).
- Build the `Execution Monitor` loop with collision detection and timeout logic.
- Validate baseline A\* correctness on static grids before introducing dynamics.
- **Deliverable**: Working A\* baseline with deterministic and stochastic environment modes.

#### Phase 2 — LangGraph Agent Scaffold

- Set up the LangGraph `StateGraph` with the `AgentState` schema.
- Implement the `Environment Scanner`, `A\* Planner`, and `Path Risk Evaluator` nodes (LLM integration via Anthropic API).
- Wire conditional edges for the risk threshold decision.
- Implement the `Path Healer` with waypoint detour logic.
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

- Run all four variants across all environment configurations and both grid sizes (full Monte Carlo protocol).
- Collect and aggregate metrics; generate plots for success rate curves, PER distributions, and learning curves.
- Run ablation comparisons; apply statistical significance tests.
- **Deliverable**: Complete results tables and figures ready for the paper.

#### Phase 6 — Paper & Repository Finalization

- Write the full IEEE-format paper following the deliverable structure above.
- Finalize the repository with a README, example run scripts, configuration files, and evaluation logs.
- **Deliverable**: Submitted PDF and GitHub repository.

#### Technology Stack

| Component            | Technology                                                     |
| -------------------- | -------------------------------------------------------------- |
| Agent Orchestration  | LangGraph (Python)                                             |
| LLM Backend          | Anthropic API (`claude-haiku-4-5` for evaluator/reflector)     |
| Grid Environment     | NumPy + custom Python simulation                               |
| A\* Implementation   | Custom Python (networkx optional)                              |
| Memory Store         | JSON log (Phase 3); optionally ChromaDB for semantic retrieval |
| Experiment Runner    | Python scripts with seed-controlled random                     |
| Visualization        | Matplotlib / Seaborn                                           |
| Statistical Analysis | SciPy                                                          |

---

### 8. Additional Considerations

#### Ethical and Safety Reflections

- The system's LLM component introduces **non-determinism**: the same grid state may yield different risk assessments across runs, which is worth analyzing as a form of bounded rationality rather than a flaw.
- **Interpretability**: The natural-language justifications produced by the Path Risk Evaluator and Reflection Node are a core artifact — they provide a human-readable audit trail of agent decisions, addressing the "black box" criticism of LLM-based agents.
- **Scope Limitation**: This system is a simulation study. Before applying similar agentic oversight to real-world robotics or autonomous vehicles, significantly more rigorous safety validation would be required.
- **Cost Awareness**: LLM API calls carry financial and latency costs. The design intentionally gates LLM invocation behind risk thresholds and uses a lightweight model (`haiku`) to make the overhead tractable at simulation scale.

#### Anticipated Challenges

- Prompt engineering for the Path Risk Evaluator to produce consistent, structured risk scores across diverse grid configurations.
- Balancing memory retrieval relevance vs. noise — overly broad memory retrieval may introduce false priors.
- Ensuring the reflection loop does not overfit to recent episodes at the expense of generalizable patterns.
