# agent-core Architecture

`agent-core` is a bounded orchestration engine for autonomous LLM-driven reasoning. The architecture follows clean/hexagonal boundaries so each major subsystem is independently replaceable and testable.

## Design Goals
- Deterministic loop control
- Strict bounded memory and context
- Planner/executor isolation
- Tool gating through policy and sandbox
- Evidence-centric state evolution
- Profile-driven domain specialization without core mutation

## Layered Architecture

```mermaid
flowchart TB
    subgraph L1["Layer 1: Orchestration Engine"]
      LC["LoopController"]
      TE["TerminationEngine"]
      SD["StagnationDetector"]
    end

    subgraph L2["Layer 2 + 6 + 7: State & Evidence & Memory"]
      AS["AgentState"]
      ST["StackTree"]
      HY["Hypothesis Set"]
      EG["EvidenceGraph"]
      MM["MidTermMemory"]
      LM["LongTermStoragePort"]
    end

    subgraph L3["Layer 3: Context Builder"]
      CB["ContextBuilder"]
      TB["TokenBudget"]
      SP["SummarizationPolicy"]
    end

    subgraph L4["Layer 4: Decision Engine"]
      DE["DecisionEngine"]
    end

    subgraph L5["Layer 5: Tool Sandbox"]
      TR["ToolRegistry"]
      TP["ToolPolicyEnforcer"]
      SB["ToolSandbox"]
    end

    subgraph L8["Layer 8: Profile Adapter"]
      PI["ProfileInterface"]
      PM["PhaseManager"]
    end

    subgraph L9["Layer 9: Observability"]
      MC["MetricsCollector"]
      LG["Structured Logger"]
    end

    PL["Planner"]
    EX["Executor"]
    LLM["LLMAdapter"]

    LC --> CB --> PL
    LC --> EX
    PL --> LLM
    EX --> LLM
    LC --> DE
    DE --> TP --> SB --> TR
    DE --> SP
    DE --> EG
    DE --> LM
    LC --> TE
    LC --> SD
    LC --> AS
    LC --> MM
    LC --> MC
    LC --> LG
    PM --> PI
    LC --> PM
```

## Deterministic Control Loop

```mermaid
sequenceDiagram
    participant LC as LoopController
    participant Ctx as ContextBuilder
    participant Pln as Planner
    participant Exe as Executor
    participant Dec as DecisionEngine
    participant Tool as Tool/Sandbox
    participant Mem as MidTermMemory
    participant Term as TerminationEngine
    participant Obs as Metrics

    LC->>Ctx: Build bounded context
    Ctx-->>LC: Structured payload + token count
    LC->>Pln: plan(context)
    Pln-->>LC: PlannerOutput
    LC->>Exe: propose(context, objective)
    Exe-->>LC: ExecutorProposal
    LC->>Dec: evaluate_and_execute(proposal)
    Dec->>Tool: validate policy + sandbox + execute
    Tool-->>Dec: raw payload
    Dec->>Mem: summary + evidence node (no raw in working set)
    Dec-->>LC: DecisionOutcome
    LC->>Term: evaluate(termination criteria)
    Term-->>LC: continue/terminate
    LC->>Obs: emit iteration metrics + transition logs
```

## State Boundaries

```mermaid
flowchart LR
    subgraph WM["Short-Term Working Memory (LLM-visible)"]
      G["Goal"]
      P["Phase"]
      A["Active Node"]
      H["Top Hypotheses"]
      E["Top Evidence Summaries"]
      T["Latest Tool Summary"]
    end

    subgraph MT["Mid-Term Structured Memory"]
      EG2["EvidenceGraph"]
      HY2["Hypotheses"]
      ST2["StackTree"]
      DH["Decision History"]
      IS["Iteration Summaries"]
    end

    subgraph LT["Long-Term Storage"]
      RAW["Raw Tool Outputs"]
      SNAP["Snapshots"]
      LOGS["Raw Reasoning Logs"]
    end

    MT -->|Selective ranking| WM
    LT -->|Pointers only| MT
```

## Core Safety Invariants
- Context budget is hard-capped by `TokenBudget` and trimmed by priority.
- Evidence growth is bounded by `EvidenceGraph.max_nodes` with deterministic pruning.
- Stack depth is bounded by `StackTree.max_depth`; collapse logic handles drift/stagnation.
- Executor proposals cannot mutate phase/hypotheses/stack directly.
- Every tool call is validated (`ToolRegistry` + schema), authorized (`ToolPolicy`), and sandboxed.
- Raw tool output is persisted to long-term storage and replaced with structured summaries.
- Every iteration emits metrics and explicit state transition logs.

## Extension Points
- Profiles: implement `ProfileInterface` and plug into `PhaseManager`.
- LLM providers: implement `LLMAdapter`, register in `ModelRegistry`.
- Storage backends: implement storage ports (`StorageBackend`, `LongTermStoragePort`).
- Multi-agent: share `EvidenceGraph` and memory contracts while adding orchestrator roles.
