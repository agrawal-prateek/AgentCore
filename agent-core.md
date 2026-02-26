# 📘 agent-core.md
Autonomous Agent Orchestration Engine
Version: 2.0 (Hardened Architecture)
Status: Production-Grade Specification

---

# 1. PURPOSE

agent-core is a deterministic, stateless, extensible orchestration engine for building autonomous LLM-driven agents.

It is domain-agnostic.

It is designed to:
- Sustain 100+ iterative loops
- Maintain flat context size
- Prevent reasoning drift
- Bound memory growth
- Enable multi-agent expansion
- Support system-control agents safely
- Provide auditability and replay

agent-core is the brain.
Profiles provide domain specialization.

---

# 2. DESIGN GUARANTEES

The engine guarantees:

1. Constant LLM context size across iterations
2. Bounded memory growth
3. Deterministic loop control
4. Explicit state transitions
5. Tool isolation
6. Evidence graph normalization
7. Controlled exploration
8. Observability at every level
9. Extensibility without architectural rewrite

---

# 3. ARCHITECTURAL LAYERS

Layer 1: Orchestration Engine
Layer 2: State Manager
Layer 3: Context Builder
Layer 4: Decision Engine
Layer 5: Tool Execution Sandbox
Layer 6: Evidence Graph Engine
Layer 7: Memory Stratification Layer
Layer 8: Profile Adapter
Layer 9: Observability & Policy Enforcement

Each layer is independently replaceable.

---

# 4. MEMORY STRATIFICATION MODEL

Memory is divided into strict tiers.

## 4.1 Short-Term Working Memory (LLM-visible)

Contains only:
- Current goal
- Current phase
- Active stack node
- Parent summaries (bounded)
- Top-N hypotheses
- Top-K relevant evidence summaries
- Current tool result summary

Hard token cap enforced.

Never accumulates transcripts.

---

## 4.2 Mid-Term Structured Memory

Contains:
- Full evidence graph
- All hypotheses
- Stack tree
- Phase summaries
- Decision history
- Iteration summaries

Not injected into LLM wholesale.
Queried selectively by Context Builder.

---

## 4.3 Long-Term Storage

Contains:
- Full tool outputs
- Full reasoning text
- Raw logs
- Raw code
- Snapshots of state
- Metrics

Never injected into LLM unless explicitly requested.

---

## 4.4 Cold Archive (Optional)

Used for:
- Cross-case learning
- Offline training
- Pattern mining

Not used during active loop.

---

# 5. STATE MODEL (STRICT CONTRACT)

All state objects are strongly typed.

## 5.1 Root State

AgentState:
    id
    goal
    status (running | paused | completed | failed)
    current_phase
    iteration_count
    branch_depth
    stagnation_counter
    exploration_score
    exploitation_score
    config_snapshot

    stack_tree_id
    evidence_graph_id
    hypothesis_set_id

    working_set_id
    decision_log_id
    summary_index_id
    metrics_id

No free-form fields allowed.

---

## 5.2 Stack Tree

Represents DFS exploration structure.

StackNode:
    id
    objective
    parent_id
    depth
    branch_score
    status (open | exhausted | validated | abandoned)
    summary
    created_at
    closed_at

Rules:
- Max depth configurable
- Auto-collapse if stagnation detected
- Parent summaries bounded to 1-2 lines

---

## 5.3 Hypothesis Model

Hypothesis:
    id
    description
    confidence_score (0-1)
    status (candidate | validated | refuted | stale)
    supporting_evidence_ids
    refuting_evidence_ids
    last_updated_iteration

Confidence recalculated via deterministic scoring rules.

---

## 5.4 Evidence Graph

EvidenceNode:
    id
    type (log | code | metric | db | inference | summary)
    source_reference
    summary
    raw_pointer
    relevance_score
    weight
    created_iteration

Relationships:
    supports
    contradicts
    derived_from
    correlates_with

Graph Constraints:
- Maximum node count threshold
- Auto-pruning of low-weight stale nodes
- Deduplication hashing
- Merge policy for similar nodes

---

# 6. DECISION ENGINE

LLM never directly executes tools.

Decision Engine mediates all LLM outputs.

Steps:
1. Planner produces next objective.
2. Executor proposes tool call.
3. Decision Engine validates:
   - Tool exists
   - Arguments valid
   - Phase permits action
   - Risk policy passes
4. Execution allowed or rejected.

If rejected:
- Re-plan triggered.

---

# 7. PLANNER / EXECUTOR REFINEMENT

## 7.1 Planner Responsibilities

- Select branch to explore
- Decide phase transition
- Promote or demote hypotheses
- Trigger synthesis

Planner sees only structured summaries.

Planner output schema is strict:
    next_objective
    target_branch_id
    phase_transition (optional)
    reasoning_summary
    termination_flag

---

## 7.2 Executor Responsibilities

- Select tool
- Provide arguments
- Estimate expected outcome

Executor cannot change:
- Hypotheses directly
- Phase directly
- Stack structure directly

Executor proposals are validated by Decision Engine.

---

# 8. CONTEXT BUILDER (DYNAMIC STRATEGY)

Context builder constructs minimal context using priority ranking.

Priority order:
1. Goal
2. Current phase rules
3. Active branch summary
4. Top hypotheses
5. Most relevant evidence
6. Latest tool result summary

Dynamic trimming rules:
- Drop lowest relevance evidence
- Compress older summaries
- Cap branch ancestry to N

Context size constant across loops.

---

# 9. STAGNATION & DRIFT CONTROL

Engine monitors:

- No new evidence discovered for N iterations
- Repeated similar tool calls
- Hypothesis confidence unchanged
- Excessive branch depth

If triggered:
- Force re-plan
- Force phase shift
- Collapse branch
- Trigger synthesis attempt

Prevents infinite loops.

---

# 10. EXPLORATION vs EXPLOITATION CONTROL

Engine tracks:

exploration_score
exploitation_score

If exploration too high:
- Restrict new branches
- Focus validation

If exploitation too high:
- Force new branch discovery

Balances reasoning behavior.

---

# 11. TOOL SANDBOX & RISK MODEL

Each tool has:

ToolPolicy:
    permission_level
    risk_score
    requires_confirmation
    max_invocations
    allowed_phases

For system-control agents:
- Read-only mode
- Safe-mode enforcement
- Approval callbacks
- Resource throttling

---

# 12. SUMMARIZATION POLICY (MANDATORY)

After every tool execution:

1. Raw output stored.
2. Automatic structured summarization.
3. Entity extraction.
4. Evidence node creation.
5. Compression scoring.

No raw output persists in working memory.

---

# 13. TERMINATION ENGINE

Termination occurs when:

- Validated hypothesis confidence > threshold
- Planner signals termination
- Iteration limit reached
- Stagnation threshold exceeded
- Risk boundary crossed

Termination always produces:
- Final synthesis
- Evidence mapping
- Confidence score
- Decision trace

---

# 14. REPRODUCIBILITY MODE (OPTIONAL)

If enabled:

- Deterministic sampling
- Fixed temperature
- Decision trace logging
- Seed tracking

Allows replay of investigation.

---

# 15. MULTI-AGENT FUTURE SUPPORT

State designed to allow:

- Parallel planners
- Critic agent
- Verification agent
- Consensus engine

All share Evidence Graph.

---

# 16. OBSERVABILITY & METRICS

Metrics tracked:

- Token usage per iteration
- Context size
- Evidence growth rate
- Branch growth rate
- Hypothesis churn rate
- Tool latency
- Loop duration

Enables health monitoring.

---

# 17. FAILURE HANDLING

Engine handles:

- Invalid LLM output
- Tool timeout
- Tool schema violation
- Hallucinated tool
- Phase deadlock
- State corruption

Recovery strategy configurable.

---

# 18. CONFIGURATION MODEL

AgentConfig:
    max_iterations
    max_depth
    max_evidence_nodes
    stagnation_threshold
    exploration_bias
    token_budget
    planner_model
    executor_model
    reproducibility_mode
    safety_mode

Profiles can override defaults.

---

# 19. PROFILE INTERFACE CONTRACT

Each profile must provide:

- Phase definitions
- Tool registry
- System prompt template
- Completion criteria
- Domain constraints
- Optional scoring rules

Profiles cannot:
- Modify engine internals
- Bypass decision engine
- Inject unbounded memory

---

# 20. HARD GUARANTEES

The engine ensures:

- Context never grows unbounded
- Evidence graph bounded
- Stack depth bounded
- Loop deterministic
- Tool calls validated
- Reasoning drift minimized

---

# 21. NON-GOALS

agent-core does NOT:
- Contain domain prompts
- Contain business logic
- Contain DB logic
- Contain UI logic
- Contain search logic

---

# 22. DESIGN PHILOSOPHY

The LLM is a reasoning module.

The engine is the governor.

The state is the source of truth.

The loop is deterministic.

The context is rebuilt every iteration.

---

# FINAL POSITIONING

agent-core is not a chatbot framework.

It is a bounded cognitive operating system
for autonomous reasoning agents.

It is designed to:
- Scale
- Persist
- Extend
- Survive production
- Control complex agentic workflows safely