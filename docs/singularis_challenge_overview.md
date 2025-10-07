# 🧩 SINGULARIS: Reforming Scientific Publishing

**Mission:** Change how scientists interact with knowledge.  
**Goal:** Build a cost-efficient AI solution to represent research papers as interconnected knowledge graphs.

---

## 🧠 CENTRAL IDEA

Our central idea is that **the minimal publishable and citable unit should be smaller than a scientific paper**.

It could be one:
- Hypothesis
- Experiment
- Method
- Result
- Dataset
- Analysis

### The Vision

To migrate the existing data to a new system, we will annotate the corpus of research papers with this new format. With the use of AI, papers will be represented as **graphs consisting of elements connected with meaningful links**.

### Core Elements

- **Hypothesis** - Scientific assumptions to be tested
- **Experiment** - Designed procedures to test hypotheses
- **Method** - Techniques and approaches used
- **Result** - Obtained data and findings
- **Dataset** - Collections of data used
- **Analysis** - Statistical and computational analyses

---

## 🔗 KNOWLEDGE GRAPH CONSTRUCTION

### STEP 1: Individual Graphs

Connect individual graphs to build a knowledge graph representing decades of scientific development — with the flow of thought, breakthroughs, and failures.

### STEP 2: Better Metrics

Provide much better metrics for scientific output that would characterize scientists based on this meaningful graph, rather than papers connected only by references.

### STEP 3: True Impact

The position of scientists will not be influenced by hype, journal reputation, or political weight.

**Only the contribution to the ideas and results that people use will be accounted for.**

---

## 📜 MISSION STATEMENT

The Singularis mission is to reform how scientists interact with knowledge.

### The Problem

Modern scientific literature is outdated — the **IMRAD format** (Introduction, Method, Results, Discussion) comes from a time when mail delivered research.

### The Solution

Singularis will modernize and accelerate this process to correspond to today's tempo and boost the efficiency of both scientists and computers.

---

## ❓ WHY COULDN'T IT BE DONE BEFORE?

Scientific funding, promotions, and reputation are tied to journal prestige and impact factors. Changing the format risks disrupting these incentives — **it's like changing the wheels of a moving vehicle**.

### The Singularis Strategy

> Install new wheels on the vehicle, transfer traction to them, then remove the old ones.

---

## ⚙️ CURRENT CHALLENGE

Significant progress has already been made using LLM-based algorithms, but these technologies are **expensive and not scalable** for the entire corpus of scientific data.

### The Challenge for Teams

**Make the algorithm more computationally efficient.**

---

## 🧩 CHALLENGE DETAILS

The task is to refine the LLM-based technology and build a **cost-efficient knowledge extraction system**, allowing for the analysis of **50 million research papers**.

### Required Elements Structure

Each paper should be decomposed into:

1. **Input Fact** - Established knowledge entering the study
2. **Hypothesis** - Proposed explanation or prediction
3. **Experiment** - Designed test of the hypothesis
4. **Technique** - Methods and tools employed
5. **Result** - Data and observations obtained
6. **Dataset** - Data collections used or generated
7. **Analysis** - Statistical/computational processing
8. **Conclusion** - Interpretations and implications

### Relationships

Each element must have **accurate representation of relationships** between elements:
- Hypothesis → tested by → Experiment
- Result → analyzed using → Analysis
- Conclusion → based on → Result
- Method → applied in → Experiment

---

## 🧮 ACCEPTABLE APPROACHES

| Approach | Description | Status |
|----------|-------------|--------|
| 🧠 **Pure Algorithmic** | Regular expressions and rules without LLMs | ✅ Welcome |
| ⚡ **Hybrid Approach** | LLM + algorithms/regex | ✅ **Recommended** |
| 🚫 **LLM-only** | Pure LLM solution | ❌ Not suitable (cost, inference time) |

---

## 📈 EVALUATION CRITERIA

### Primary Metrics

1. **Code's accuracy:** Correct graph construction
2. **Computational efficiency:** Cost and performance optimization

### Detailed Evaluation

#### 1. Completeness / Accuracy (25%)
- **Precision:** % of correctly extracted elements
- **Recall:** % of all elements found
- **F1 Score:** Harmonic mean of precision and recall
- Measured for each element type and relation

#### 2. Correctness (included in accuracy)
- Reference and citation extraction accuracy
- Proper entity resolution
- Correct relationship identification

#### 3. Robustness (25%)
- Handling of different article formats (PDF, HTML, XML)
- Stability across different journals and writing styles
- Error handling and recovery

#### 4. Cost Analysis (25%)
- **CPU/GPU hours:** Computational resources used
- **$ per paper:** Total processing cost
- **Tokens used:** API consumption metrics
- Scalability projection to 50M papers

#### 5. Performance (25%)
- **Throughput:** Papers processed per hour
- **Latency:** Response time on fixed hardware
- **Parallelization:** Ability to scale horizontally

---

## ⭐ BONUS POINTS

Extra credit will be awarded to:

1. **Algorithmic or hybrid solutions** that achieve significant cost reduction while maintaining comparable quality to LLM-only systems

2. **Teams that improve the conceptual framework** or bring additional innovation:
   - Novel extraction techniques
   - Better relationship detection
   - Improved graph structures
   - Creative cost optimization strategies

---

## 🎯 FINAL OBJECTIVE

Develop a computationally efficient pipeline capable of:

✅ **Extracting and structuring** research paper content  
✅ **Building scalable knowledge graphs** from millions of papers  
✅ **Measuring true scientific impact** based on contribution, not publication prestige  
✅ **Processing cost:** < $0.05 per paper (target)  
✅ **Throughput:** > 100 papers/hour (target)  

---

## 🚀 SUCCESS METRICS

### Target Benchmarks

- **Precision:** ≥ 85%
- **Recall:** ≥ 80%
- **F1-score:** ≥ 82%
- **Cost:** < $0.05 per paper
- **Speed:** > 100 papers/hour
- **Scalability:** Projectable to 50M papers

---

## 💡 KEY INSIGHTS

### What Makes This Hard?

1. **Scale:** 50 million papers to process
2. **Cost:** LLM inference is expensive at scale
3. **Accuracy:** Must maintain high precision and recall
4. **Diversity:** Different formats, journals, writing styles
5. **Relationships:** Complex connections between elements

### What Makes This Important?

1. **Scientific Progress:** Accelerate knowledge discovery
2. **True Merit:** Measure impact by contribution, not prestige
3. **Accessibility:** Make research more discoverable
4. **Efficiency:** Save researchers' time
5. **Innovation:** Enable new ways of scientific collaboration

---

## 📝 SUBMISSION REQUIREMENTS

### Required Deliverables

1. **Working code** - Deployed and accessible
2. **Documentation** - Architecture, approach, metrics
3. **Demo video** - 3-5 minutes showing the system
4. **Cost analysis** - Detailed breakdown
5. **Performance metrics** - Precision, recall, throughput, cost

### Deadline

**October 22, 2025 at 11:59 PM PT** (Code Freeze)

---

**Good luck building the future of scientific publishing! 🚀**

*Last updated: October 7, 2025*