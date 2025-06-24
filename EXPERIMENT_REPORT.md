# Nutrition Agent Experiment Report

## Summary

This report summarizes my approaches, roadblocks, and findings from developing an AI-powered nutrition and recipe planning agent. The project explored single-agent architectures with tool-calling capabilities, achieving moderate success in constraint satisfaction (>96%) but struggling with nutrition accuracy (60-72%). Key insights include the effectiveness of tool-based validation and the critical need for better planning algorithms.

## Approaches Tried

### 1. Single-Agent Architecture with Tool Calling

**Implementation**: Base agent using OpenAI's function calling to query FatSecret API for nutrition data.

**Results**:

- ✅ Clean separation between reasoning and data retrieval
- ✅ Reduced hallucination of nutrition values
- ❌ Sequential processing limited speed
- ❌ No parallel exploration of meal options

### 2. Iterative Refinement Loop

**Implementation**: Up to 10 iterations allowing the agent to correct mistakes based on validation feedback.

**Results**:

- ✅ Improved constraint satisfaction from 85% to 96%
- ✅ Self-correction for obvious errors
- ❌ Often stuck in local optima
- ❌ No strategic replanning, only tactical fixes

### 3. Structured Output with JSON Schema

**Implementation**: Enforced output format using OpenAI's structured generation and fallback parsing.

**Results**:

- ✅ 95% successful parsing rate
- ✅ Consistent output format
- ❌ Fragile string-based extraction as fallback
- ❌ Lost flexibility in creative meal suggestions

### 4. Multi-Model Evaluation

**Models tested**: GPT-4.1, GPT-4.1-mini, GPT-4.1-nano

**Comparative Results**:

```
Model         Overall  Nutrition  Constraints  Cost/Run  Speed
GPT-4.1       0.618    0.000      0.962        $0.12     23.2s
GPT-4.1-mini  0.765    0.602      0.973        $0.03     8.7s  ← Best overall
GPT-4.1-nano  0.701    0.235      0.980        $0.01     5.1s
```

## Major Roadblocks Encountered

### 1. Nutrition Calculation Accuracy

**Problem**: All models struggled with precise PFC ratio calculations, with GPT-4.1 completely failing (0.000 score) in one test. Note that if the maximum target error for each P/F/C is >50%, it will be clipped to 0.

**Root Cause**:

- Models attempted to mentally calculate complex nutritional combinations
- No explicit mathematical planning phase
- Accumulated errors across multiple ingredients

**Attempted Solutions**:

- ✅ Added nutrition calculator tool for validation
- ❌ Still relied on LLM for initial meal selection
- 🔄 Needed: Explicit optimization algorithm for ingredient selection w.r.t nutrition

### 2. Lack of Semantic Understanding

**Problem**: Keyword-based allergen detection missed compound ingredients.

**Example**: System didn't flag mayonnaise for egg allergies.

**Attempted Solutions**:

- ✅ Expanded keyword lists
- ❌ Still missed edge cases
- 🔄 Needed: use LLM-as-a-judge to capture ingredient relationships

### 3. Static Decision Making

**Problem**: No learning from previous runs or user feedback.

**Impact**:

- Repeated similar mistakes
- No personalization over time
- Suboptimal meal variety

### Recipe Preference Modeling

**Problem**: The current system does not adequately account for user preferences beyond basic dietary constraints. This limits the diversity of meal suggestions and often results in repetitive or unappealing options for users. While nutritional targets are met, the variety and efficiency of meal preparation (e.g., maximizing use of existing ingredients, minimizing cooking time across meals) are not optimized.

**Root Cause**:

- Lack of a robust preference learning mechanism.
- Over-reliance on a fixed set of recipes without dynamic adaptation.
- No explicit consideration of "meal flow" or ingredient reuse across multiple meals.

**Attempted Solutions**:

- ❌ Simple random selection for variety (led to uncohesive meals).
- 🔄 Needed: A sophisticated preference model that balances nutritional goals with user tastes, ingredient availability, and preparation efficiency, potentially leveraging RL techniques.

## Evaluation Methods Analysis

### Most Effective Methods

1. **Weighted Multi-Metric Scoring**

   - Balanced nutrition (30%), constraints (25%), inventory (25%), quality (20%)
   - Provided nuanced performance assessment
   - Identified specific failure modes

2. **Zero-Tolerance Constraint Checking**

   - Critical for allergen safety
   - Clear pass/fail criteria
   - Prevented dangerous recommendations

3. **Component-wise Error Analysis**
   ```python
   nutrition_errors = {
       "calories": 12.3%,
       "protein": 8.7%,
       "fat": 22.1%,  # Biggest challenge
       "carbs": 15.4%
   }
   ```

### Less Effective Methods

1. **Binary Success Metrics**: Too coarse for optimization
2. **Execution Time as Primary Metric**: Trade-off with quality wasn't linear
3. **Simple Diversity Counting**: Didn't capture meal quality or cultural appropriateness

## Smallest Effective Model

**GPT-4.1-mini** emerged as the optimal choice:

- **Performance**: 0.765 overall score (best among tested)
  - Actually, several experiments showed no significant difference in performance between GPT-4.1 and GPT-4.1-mini, and the performance was not stable.
- **Cost Efficiency**: 3.4x better than GPT-4.1 ($0.03 vs $0.12)
- **Speed**: 8.7s average
- **Reliability**: Most consistent across test scenarios

**Key Insight**: Smaller models with proper tool support outperformed larger models without tools.

## Specific Feedback Requests

### 1. Reward Shaping for Nutrition and Recipe Optimization

**Current Challenge**: Linear penalty for nutrition errors leads to acceptance of consistent 20% errors rather than attempting perfect matches.

**Question**: How would you design a reward function that encourages exploration of exact nutritional matches while maintaining meal diversity? Should we use:

- Sparse rewards (exact match or nothing)?
- Curriculum learning (gradually tightening tolerances)?
- Multi-objective RL with Pareto optimization?

### 2. Exploration vs Exploitation in Meal Planning

**Current Issue**: Agent quickly converges to "safe" meal combinations (rice + chicken repeatedly).

**Question**: What exploration strategies would work best for this combinatorial problem?

### 3. State Representation for Sequential Decisions

**Current State**: Flat representation of inventory + constraints + targets.

**Question**: How should we encode the sequential nature of meal planning?

### 4. Credit Assignment Problem

**Challenge**: Final reward depends on all 9 meals (3 days × 3 meals), making it hard to identify which specific meal choices led to success/failure.

**Question**: Should we implement some ideas like hierarchical RL (day-level and meal-level policies) or some reward decomposition with meal-specific feedback?

### 5. Constraint Satisfaction in RL

**Critical Requirement**: Zero tolerance for allergen violations.

**Question**: Best approach for hard constraints in RL?

- Lagrangian methods with dynamic penalties?
- Constrained policy optimization (CPO)?
- Shield synthesis for safety?

### 6. Multi-Agent Potential

**Proposed**: Specialist agents for nutrition, inventory, quality.

**Question**: For multi-agent RL in this domain:

- Centralized training with decentralized execution?
- Communication protocols between agents?
- How to handle credit assignment across agents?

## Next Steps

Based on experimental findings, possible improvements would be:

1. **Replace reactive planning with proactive optimization** (possibly using RL)
2. **Improve reward functions for judging ingredient attributes**
3. **Develop better state representations for sequential decisions**

The current architecture provides a solid foundation, but the lack of strategic planning and learning capabilities fundamentally limits performance. RL approaches could address both issues while maintaining the successful tool-based validation framework.
