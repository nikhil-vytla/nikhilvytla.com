# Category 1: Technical Surveys & Literature Reviews

## Overview

Technical surveys transform scattered academic literature into authoritative references that compress months of research into digestible, actionable frameworks. These posts serve as definitive guides for rapidly evolving AI/ML domains, combining comprehensive coverage with practical implementation guidance.

## Analyzed Posts

- **"Prompt Engineering"** (March 15, 2023) - 50+ citations, 8,000+ words
- **"LLM Powered Autonomous Agents"** (June 23, 2023) - 80+ citations, 12,000+ words
- **"Adversarial Attacks on LLMs"** (October 25, 2023) - 60+ citations, 10,000+ words

## Authority Building Strategy

### Definitive Opening Statements

You should establish authority through clear, canonical definitions:

> **"Prompt Engineering, also known as In-Context Prompting, refers to methods for how to communicate with LLM to steer its behavior for desired outcome without updating the model weights."**

**Why this works:**

1. **Definitional authority**: Provides the canonical definition
2. **Terminological bridge**: Links familiar terms to technical precision
3. **Clear scope**: Bounds exactly what will be covered
4. **Immediate value**: Readers know they're getting the authoritative take

**Compare with an example agents post:**

> **"In a LLM-powered autonomous agent system, LLM functions as the agent's brain, complemented by several key components..."**

**Pattern**: Accessible metaphor ("brain") + systematic promise ("key components")

### Transition Mastery

Your transitions should guide readers through complex conceptual territory:

**Sequential logic:**

> "However, few-shot can be expensive in terms of token usage and restricts the input length"

**Causal reasoning:**

> "At its core, the goal of prompt engineering is about alignment and model steerability"

**Evaluative progression:**

> "The accuracy of such a process depends on the quality of both retrieval and generation steps"

**Pattern**: Each transition summarizes the previous concept while forecasting the next analysis.

## Research Synthesis Framework

### Citation Integration Strategy

**Three citation approaches:**

**Inline authority building:**

```
Zhao et al. (2021) investigated few-shot classification and found that...
```

**Parenthetical efficiency:**

```
Chain-of-thought (CoT) prompting (Wei et al. 2022) generates...
```

**Clustered evidence:**

```
Multiple studies (Brown et al. 2020; Liu et al. 2021; Min et al. 2022) demonstrate...
```

### Citation Density Analysis

| Post Category         | Citations/1000 words | Integration Style  |
| --------------------- | -------------------- | ------------------ |
| Technical Surveys     | 12-15                | Woven throughout   |
| Deep Tutorials        | 8-10                 | Theory sections    |
| Problem Analysis      | 10-12                | Evidence-focused   |
| Conceptual Deep-Dives | 6-8                  | Framework-building |

## Information Architecture

### Hierarchical Organization

You should consistently employs progressive disclosure:

```
## Prompt Engineering
├── Basic Prompting
│   ├── Zero-shot
│   └── Few-shot
├── Advanced Techniques
│   ├── Chain-of-Thought
│   ├── Tree of Thoughts
│   └── Self-Consistency
└── Tool Integration
    ├── Retrieval Augmentation
    └── Code Generation
```

### Comparative Framework Construction

**Method comparison matrices:**

| Technique | Complexity | Token Cost | Reliability | Use Case            |
| --------- | ---------- | ---------- | ----------- | ------------------- |
| Zero-shot | Low        | Low        | Medium      | Quick tasks         |
| Few-shot  | Medium     | High       | High        | Complex reasoning   |
| CoT       | High       | High       | Very High   | Multi-step problems |

## Language Patterns & Communication

### Sentence Architecture

**Complex-compound structure (70% of sentences):**

- Primary clause: establishes core concept
- Subordinate clauses: add qualification and nuance
- Parenthetical additions: provide implementation details

**Example breakdown:**

> **"Chain-of-thought (CoT) prompting (Wei et al. 2022) generates a sequence of short sentences that mimic the reasoning process a human might have when working through the problem."**

Structure analysis:

- **Core assertion**: "CoT prompting generates sentences"
- **Citation**: "(Wei et al. 2022)" for immediate credibility
- **Function**: "that mimic the reasoning process"
- **Accessibility**: "human might have" bridges to intuition
- **Context**: "when working through the problem"

### Uncertainty Management

**Sophisticated hedging strategies:**

**Soft hedging** (maintains authority):

> "The effect of prompt engineering methods can vary a lot among models, thus requiring heavy experimentation"

**Personal positioning** (builds credibility):

> "[My personal spicy take] In my opinion, some prompt engineering papers are not worthy 8 pages long"

**Methodological skepticism** (demonstrates rigor):

> "Currently does not support tool use in a chain" [regarding Toolformer limitations]

**Function**: Acknowledges uncertainty while maintaining credibility and protecting readers from over-application.

## Tone & Voice Analysis: The Curator

### Primary Tone: Confident Without Arrogance

You should position yourself as a **synthesizer** rather than the original researcher, creating authority through comprehensiveness rather than claims of innovation.

**Voice characteristics:**

- **Trustworthy**: Provides canonical definitions and systematic taxonomies
- **Humble**: Extensive attribution to original researchers
- **Protective**: Warns readers about limitations and implementation costs
- **Occasionally personal**: Strategic injection of personality

### Strategic Voice Modulation

**Professional voice (90% of content):**

> "Recent advances in large language models have demonstrated remarkable capabilities in few-shot learning scenarios, where models can adapt to new tasks with minimal examples."

**Personal voice (10% of content - strategic placement):**

> "[My personal spicy take] In my opinion, some prompt engineering papers are not worthy 8 pages long"

**Pattern**: Just enough personality to prevent academic monotony, never enough to overshadow content.

### Authority Building Without Ego

**Build credibility through:**

1. **Comprehensive coverage**: "This survey covers 50+ recent papers on prompt engineering"
2. **Critical evaluation**: "While X shows promise, it suffers from Y limitations"
3. **Implementation wisdom**: "In practice, you'll want to consider Z factors"

**Maintain humility through:**

1. **Attribution abundance**: Extensive citing and credit-giving
2. **Limitation honesty**: "Further work is needed to address..."
3. **Reader protection**: Warning about method boundaries and costs

### Tone Comparison by Content Type

**When introducing concepts**: Most authoritative

> "Prompt engineering refers to methods for communicating with LLMs..."

**When evaluating methods**: Balanced and critical

> "While CoT prompting shows impressive results, it requires larger models and increases token costs"

**When acknowledging gaps**: Most humble

> "Current approaches have several limitations that merit further investigation"

### The "Trustworthy Expert" Formula

**Authority** (through comprehensive work) + **Humility** (through limitation acknowledgment) + **Accessibility** (through clear explanation) + **Strategic personality** (through occasional asides) = **Trustworthy expertise**

This creates a voice that readers:

- **Trust** because it's honest about limitations
- **Learn from** because it's pedagogically sophisticated
- **Enjoy** because it has just enough personality
- **Cite** because it's authoritative and comprehensive

## Code & Technical Examples

### Documentation Philosophy

```python
def chain_of_thought_prompt(question: str, examples: List[str]) -> str:
    """
    Implements Chain-of-Thought prompting for complex reasoning tasks.

    Based on Wei et al. (2022) "Chain-of-Thought Prompting Elicits
    Reasoning in Large Language Models"

    Args:
        question: Problem requiring step-by-step reasoning
        examples: Few-shot examples with explicit reasoning steps

    Returns:
        Formatted prompt with reasoning chain examples

    Example:
        >>> examples = [
        ...     "Q: Roger has 5 tennis balls. He buys 2 more cans...",
        ...     "A: Roger started with 5 tennis balls. 2 cans..."
        ... ]
        >>> prompt = chain_of_thought_prompt("How many tennis balls?", examples)
    """
    template = """
    Here are examples of step-by-step reasoning:

    {examples}

    Now solve this step-by-step:
    Q: {question}
    A: Let me think through this step by step.
    """

    return template.format(
        examples="\n\n".join(examples),
        question=question
    )
```

**Code characteristics:**

- **Academic grounding**: Direct paper citations in docstrings
- **Usage guidance**: Concrete examples with expected outputs
- **Implementation clarity**: Variable names match academic terminology
- **Extensibility**: Structured for modification and expansion

### Prompt Template Sophistication

```python
# Agent prompt template from autonomous agents post
AGENT_PROMPT_TEMPLATE = """
You are an AI assistant with access to the following tools:
{tool_descriptions}

Your task: {user_task}

Think step by step:
1. What information do I need?
2. Which tools can provide this information?
3. How should I sequence the tool calls?
4. What might go wrong and how can I handle it?

Begin your response with your reasoning, then execute the tools.
"""
```

**Template design principles:**

- **Structured reasoning**: Explicit thinking steps
- **Tool integration**: Dynamic tool injection
- **Error anticipation**: Proactive problem-solving
- **Transparency**: Reasoning visibility for debugging

## Multi-Level Audience Engagement

### Accessibility Laddering

**Level 1 - Executive summary** (for busy practitioners):

> "Prompt engineering refers to methods for communicating with LLMs to steer behavior"

**Level 2 - Technical detail** (for implementers):

> "Chain-of-thought prompting generates intermediate reasoning steps by providing few-shot examples that demonstrate explicit reasoning"

**Level 3 - Research depth** (for researchers):

> "CoT prompting effectiveness emerges at scale (>100B parameters) and correlates with model performance on arithmetic and commonsense reasoning benchmarks"

### Reader Psychology Management

**Cognitive load reduction:**

- **Visual breaks**: Strategic diagrams and code blocks
- **Chunking**: Information in digestible sections
- **Preview/review**: Section summaries and transitions

**Engagement maintenance:**

- **Personal asides**: "[My personal spicy take]" breaks academic monotony
- **Practical applications**: Each concept linked to implementation
- **Future orientation**: "Open questions" sections maintain curiosity

## Visual Integration Strategy

### Diagram Types and Functions

**Architectural diagrams** (system-level understanding):

```
[User Query] → [Planning Module] → [Memory Module] → [Tool Use] → [Output]
                     ↓                   ↓              ↓
               [Sub-task Analysis]  [Context Retrieval] [API Calls]
```

**Process flows** (sequential understanding):

```
Input → Tokenization → Embedding → Attention → Feed-Forward → Output
         ↓              ↓           ↓            ↓
    [Vocabulary]   [Position]   [Multi-Head]  [Non-linear]
```

**Comparative tables** (decision support):

```
| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| Zero-shot | Simple, fast | Less reliable | Quick tests |
| Few-shot | More reliable | Token expensive | Production |
```

## Research Translation Methodology

### From Paper to Practice

**Stage 1: Concept extraction**

- Identify core innovation from academic paper
- Strip away experimental specifics
- Focus on generalizable principles

**Stage 2: Implementation translation**

- Convert mathematical formulations to code
- Provide practical parameter guidance
- Address real-world constraints

**Stage 3: Integration guidance**

- Show how technique fits existing workflows
- Identify complementary methods
- Highlight potential pitfalls

### Translation Example: Chain-of-Thought

**Original paper finding:**

> "CoT prompting improves performance on arithmetic reasoning tasks when models exceed 100B parameters"

**Example translation:**

> "Chain-of-thought prompting works by providing examples that show explicit reasoning steps. This technique is most effective with larger models and benefits tasks requiring multi-step reasoning. Implementation involves crafting few-shot examples that demonstrate the thinking process you want the model to follow."

**Added value:**

- **Actionable guidance**: How to implement
- **Boundary conditions**: When it works/doesn't work
- **Practical wisdom**: What examples to craft

## Quality Assessment Framework

### Evidence Hierarchy

**Tier 1**: Controlled experiments with statistical significance
**Tier 2**: Observational studies with clear methodology
**Tier 3**: Case studies and anecdotal reports
**Tier 4**: Theoretical arguments without empirical support

### Critical Evaluation Pattern

**Balanced assessment example:**

> "While CoT prompting shows impressive results on reasoning tasks, it requires larger models, increases token costs, and may not transfer to all domains. The technique works best when you can provide high-quality reasoning examples and have sufficient computational resources."

**Analysis:**

- **Strengths acknowledgment**: "impressive results"
- **Limitation specification**: "requires larger models"
- **Cost consideration**: "increases token costs"
- **Boundary conditions**: "may not transfer to all domains"
- **Implementation guidance**: "high-quality reasoning examples"

## Practical Implementation Guide

### Technical Survey Heading Structure Example: "LLM Powered Autonomous Agents"

```markdown
# LLM Powered Autonomous Agents

# Table of Contents # Creates navigation roadmap, signals comprehensive coverage

# Agent Framework Overview # Establishes definitional foundation immediately

## What are LLM-powered agents? # Direct question creates engagement

## What's the basic framework? # Frames complex concept simply

## What's the survey's scope? # Sets clear boundaries and expectations

# Planning # Creates natural progression from basic to complex

## Task Decomposition # Specific component everyone can understand

### Chain of thought # Familiar starting point

### Tree of thoughts # Natural extension

### LLM+P # Advanced variant

## Self-Reflection # Higher-order cognitive ability

### ReAct # Concrete implementation

### Reflexion # Academic reference point

# Memory # Parallel structure maintains cognitive organization

## Types of memory # Establishes taxonomy immediately

### Sensory memory # Draws on familiar psychological concepts

### Short-term memory # Creates accessible mental model

### Long-term memory # Complete categorization

## Memory structures # Implementation-focused subdivision

### Maximum Inner Product Search (MIPS) # Technical precision

### Memory architectures # Broader implementation patterns

# Tool use # Action-oriented section title

## Tool selection # Key challenge identification

## Tool execution # Implementation details

## MRKL # Specific system example

### ReAct # Cross-referenced from Planning section

### WebGPT # Real-world application

## HuggingGPT # Another concrete system

# Case study: Scientific discovery agents # Concrete application example

## ChemCrow # Domain-specific implementation

## Embodied agents # Physical interaction cases

### Interactive language agents # Subcategory refinement

## Generative Agents # Social simulation angle

# Proof-of-concept examples # Practical validation section

## Evaluation # Critical assessment component

## Challenges # Honest limitation acknowledgment

# Citation # Standard academic closure
```

**What makes these headings effective:**

- **Progressive complexity**: Starts with "What is?" and builds to advanced implementations
- **Parallel structure**: Planning/Memory/Tool use create logical equivalence
- **Concrete examples**: Every abstract concept paired with specific systems (ReAct, WebGPT)
- **Cross-references**: ReAct appears in both Planning and Tool use, showing interconnections
- **Psychological grounding**: Memory types use familiar cognitive science frameworks
- **Real-world validation**: Case studies and proof-of-concept provide practical anchoring
- **Honest assessment**: Dedicated sections for challenges and limitations build credibility

### Survey Writing Process

1. **Literature review**: Comprehensive coverage of domain
2. **Concept extraction**: Identify core innovations and principles
3. **Framework construction**: Organize into coherent taxonomy
4. **Implementation translation**: Bridge theory to practice
5. **Critical evaluation**: Assess strengths, limitations, trade-offs
6. **Future orientation**: Identify research gaps and directions

### Key Writing Techniques

1. **Authority establishment**: Clear definitions and comprehensive scope
2. **Progressive disclosure**: Simple to complex concept introduction
3. **Multi-perspective analysis**: Compare and contrast approaches
4. **Evidence integration**: Weave citations throughout narrative
5. **Practical bridges**: Connect abstract concepts to implementation
6. **Honest assessment**: Acknowledge limitations and uncertainties

### Communication Strategies

1. **Multi-level engagement**: Content accessible to different reader types
2. **Visual integration**: Diagrams and tables support text
3. **Code examples**: Direct implementation guidance
4. **Personal voice**: Occasional insights and opinions
5. **Forward-looking**: Open questions and future directions

## Field Impact Indicators

### Acceleration Impact

- Practitioners implement without reading 50+ papers
- Concepts spread faster through accessible synthesis
- Implementation barriers reduced through practical guidance

### Quality Control

- Critical evaluation prevents cargo-cult adoption
- Limitation acknowledgment enables informed decisions
- Systematic comparison enables better method selection

---

Analysis from LW's posts.
