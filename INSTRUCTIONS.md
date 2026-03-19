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
