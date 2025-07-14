# Awesome-RAG-Reasoning

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="Assets/RAG_Reasoning.png" alt="RAG and Reasoning System Overview" width="600"/>
</p>

A curated collection of resources, papers, tools, and implementations that bridge the gap between **Retrieval-Augmented Generation (RAG)** and **Reasoning** in Large Language Models and Agents. This repository brings together traditionally separate research domains to enable more powerful Agentic AI systems.

## 📖 Introduction

**Retrieval-Augmented Generation (RAG)** has emerged as a powerful paradigm that combines the strengths of large language models with external knowledge retrieval. By augmenting language models with relevant information from external sources, RAG systems can provide more accurate, up-to-date, and factual responses while maintaining the generative capabilities of modern LLMs.

**Reasoning** has recently gained significant popularity as a complementary approach to enhance LLM performance. Reasoning techniques focus on improving the model's ability to process information, perform logical analysis, and arrive at conclusions through structured thinking processes. These methods enable LLMs to tackle complex problems that require multi-step inference, causal understanding, and systematic problem-solving.

Although RAG and Reasoning address different aspects of the model's capabilities. **they have been developed largely independently**, with separate research communities, methodologies, and evaluation benchmarks:

**🔍 RAG Community**: Focused on knowledge retrieval, document processing, and factual grounding
- **Key Mechanism**: Query → Retrieve → Augment → Generate
- **Use Cases**: Question answering, fact verification, domain-specific applications
- **Limitations**:
  - May retrieve irrelevant or inaccurate information
  - Limited by the quality and coverage of external knowledge bases

**🧠 Reasoning Community**: Focused on logical inference, step-by-step thinking, and problem decomposition
- **Key Mechanism**: Problem → Decompose → Reason → Synthesize
- **Use Cases**: Mathematical reasoning, logical puzzles, strategic planning, causal analysis
- **Limitations**:
  - Often hallucinates or mis-grounds facts
  - Struggles with up-to-date or domain-specific information

**This repository serves as a comprehensive collection that bridges these traditionally separate domains**, providing resources for researchers and practitioners interested in combining the strengths of both approaches.

<p align="center">
  <img src="Assets/Framework.png" alt="RAG and Reasoning Framework" width="600"/>
</p>

*The **Reasoning-Enhanced RAG** methods and **RAG-Enhanced Reasoning** methods represent **one-way** enhancements. In contrast, the **Synergized RAG-Reasoning System** performs reasoning and retrieval **iteratively**, enabling mutual enhancements.*


### Why RAG + Reasoning?
Large Language Models (LLMs) serve as the foundation for modern AI systems, but they face significant limitations in both knowledge access and reasoning capabilities. 
While RAG excels at providing factual knowledge and reasoning excels at logical processing, real-world problems often require both capabilities simultaneously. Complex queries demand not just access to relevant information, but also the ability to reason through that information systematically.

**Real-World Impact**: This combination enables AI systems to tackle complex problems that require both knowledge retrieval and sophisticated reasoning, such as scientific research, legal analysis, medical diagnosis, and strategic planning.

<p align="center">
  <img src="Assets/Taxonomy.png" alt="RAG and Reasoning Taxonomy" width="600"/>
</p>

### What This Repository Covers

### 📚 Research Papers & Frameworks
Latest academic publications and open-source implementations

#### [Reasoning-Enhanced RAG](#reasoning-enhanced-rag)
- [Retrieval Optimization](#retrieval-optimization)
- [Integration Enhancement](#integration-enhancement) 
- [Generation Enhancement](#generation-enhancement)

#### [RAG-Enhanced Reasoning](#rag-enhanced-reasoning)
- [External Knowledge Retrieval](#external-knowledge-retrieval)
  - [Knowledge Base](#knowledge-base)
  - [Web Retrieval](#web-retrieval)
  - [Tool Using](#tool-using)
- [In-context Retrieval](#in-context-retrieval)
  - [Prior Experience](#prior-experience)
  - [Example or Training Data](#example-or-training-data)

#### [Synergized RAG and Reasoning](#synergized-rag-and-reasoning)
- [Reasoning Workflow](#reasoning-workflow)
  - [Chain-based](#chain-based)
  - [Tree-based](#tree-based)
  - [Graph-based](#graph-based)
    - [Walk-on-Graph](#walk-on-graph)
    - [Think-on-Graph](#think-on-graph)
- [Agentic Orchestration](#agentic-orchestration)
  - [Single-Agent](#single-agent)
    - [Prompting](#prompting)
    - [Supervised Fine-Tuning](#supervised-fine-tuning)
    - [Reinforcement Learning](#reinforcement-learning)
  - [Multi-Agent](#multi-agent)

---

### 📊 Benchmarks & Datasets
Evaluation benchmarks and training/Testing datasets

*[Will update soon]*

---

## Reasoning-Enhanced RAG
### Retrieval Optimization
- (AAAI 2025) **MaFeRw: Query Rewriting with Multi-Aspect Feedbacks for Retrieval-Augmented Large Language Models** [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/34732) [[Code]](https://github.com/yjEugenia/MaFeRw) ![GitHub Repo stars](https://img.shields.io/github/stars/yjEugenia/MaFeRw?style=social)
- (ArXiv 2025) **Collab-RAG: Boosting Retrieval-Augmented Generation for Complex Question Answering via White-Box and Black-Box LLM Collaboration** [[Paper]](https://arxiv.org/abs/2504.04915) [[Code]](https://github.com/ritaranx/Collab-RAG/) ![GitHub Repo stars](https://img.shields.io/github/stars/ritaranx/Collab-RAG?style=social)
- (ArXiv 2025) **DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning** [[Paper]](https://arxiv.org/abs/2503.00223) [[Code]](https://github.com/pat-jj/DeepRetrieval) ![GitHub Repo stars](https://img.shields.io/github/stars/pat-jj/DeepRetrieval?style=social)
- (ArXiv 2025) **Credible plan-driven rag method for multi-hop question answering** [[Paper]](https://arxiv.org/abs/2504.16787)
- (ArXiv 2025) **FIND: Fine-grained Information Density Guided Adaptive Retrieval-Augmented Generation for Disease Diagnosis** [[Paper]](https://arxiv.org/abs/2502.14614)
- (ArXiv 2025) **LLM-Independent Adaptive RAG: Let the Question Speak for Itself** [[Paper]](https://arxiv.org/abs/2505.04253) [[Code]](https://github.com/marialysyuk/External_Adaptive_Retrieval) 
- (ACL 2024) **Chain-of-Verification Reduces Hallucination in Large Language Models** [[Paper]](https://aclanthology.org/2024.findings-acl.212/)
- (EMNLP 2024) **Learning to Plan for Retrieval-Augmented Large Language Models from Knowledge Graphs** [[Paper]](https://aclanthology.org/2024.findings-emnlp.459/) [[Code]](https://github.com/zjukg/LPKG) ![GitHub Repo stars](https://img.shields.io/github/stars/zjukg/LPKG?style=social)
- (EMNLP 2024) **Retrieval and Reasoning on KGs: Integrate Knowledge Graphs into Large Language Models for Complex Question Answering** [[Paper]](https://aclanthology.org/2024.findings-emnlp.446/) [[Code]](https://github.com/Dereck0602/Retrieval-and-Reasoning-on-KGs) ![GitHub Repo stars](https://img.shields.io/github/stars/Dereck0602/Retrieval-and-Reasoning-on-KGs?style=social)
- (NAACL 2024) **Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity** [[Paper]](https://arxiv.org/abs/2403.14403) [[Code]](https://github.com/starsuzi/Adaptive-RAG) ![GitHub Repo stars](https://img.shields.io/github/stars/starsuzi/Adaptive-RAG?style=social)
- (SIGIR 2024) **Can Query Expansion Improve Generalization of Strong Cross-Encoder Rankers?** [[Paper]](https://arxiv.org/abs/2311.09175)
- (LREC-COLING 2024) **RADCoT: Retrieval-Augmented Distillation to Specialization Models for Generating Chain-of-Thoughts in Query Expansion** [[Paper]](https://aclanthology.org/2024.lrec-main.1182/) [[Code]](https://github.com/ZIZUN/RADCoT) ![GitHub Repo stars](https://img.shields.io/github/stars/ZIZUN/RADCoT?style=social)
- (ArXiv 2024) **GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning** [[Paper]](https://arxiv.org/abs/2405.20139) [[Code]](https://github.com/cmavro/GNN-RAG) ![GitHub Repo stars](https://img.shields.io/github/stars/cmavro/GNN-RAG?style=social)
- (ArXiv 2024) **RuleRAG: Rule-Guided Retrieval-Augmented Generation with Language Models for Question Answering** [[Paper]](https://arxiv.org/abs/2410.22353) [[Code]](https://anonymous.4open.science/r/RuleRAG)

### Integration Enhancement
- (ArXiv 2025) **DualRAG: A Dual-Process Approach to Integrate Reasoning and Retrieval for Multi-Hop Question Answering** [[Paper]](https://arxiv.org/abs/2504.18243)

- (EMNLP 2024) **SEER: Self-Aligned Evidence Extraction for Retrieval-Augmented Generation** [[Paper]](https://aclanthology.org/2024.emnlp-main.178/) [[Code]](https://github.com/HITsz-TMG/SEER) ![GitHub Repo stars](https://img.shields.io/github/stars/HITsz-TMG/SEER?style=social)
- (ICLR 2024) **Making Retrieval-Augmented Language Models Robust to Irrelevant Context** [[Paper]](https://openreview.net/forum?id=ZS4m74kZpH) [[Code]](https://github.com/oriyor/ret-robust) ![GitHub Repo stars](https://img.shields.io/github/stars/oriyor/ret-robust?style=social)
- (ACL 2024) **BeamAggR: Beam Aggregation Reasoning over Multi-source Knowledge for Multi-hop Question Answering** [[Paper]](https://aclanthology.org/2024.acl-long.67/)

### Generation Enhancement
- (AAAI 2025) **Improving Retrieval Augmented Language Model with Self-Reasoning** [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/34743)
- (ArXiv 2025) **RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models** [[Paper]](https://arxiv.org/abs/2503.23513)
- (ArXiv 2025) **AlignRAG: An Adaptable Framework for Resolving Misalignments in Retrieval-Aware Reasoning of RAG** [[Paper]](https://www.arxiv.org/abs/2504.14858v1) [[Code]](https://github.com/QQW-ing/RAG-ReasonAlignment) ![GitHub Repo stars](https://img.shields.io/github/stars/QQW-ing/RAG-ReasonAlignment?style=social)

- (EMNLP 2024) **Open-RAG: Enhanced Retrieval Augmented Reasoning with Open-Source Large Language Models** [[Paper]](https://aclanthology.org/2024.findings-emnlp.831/) [[Code]](https://github.com/ShayekhBinIslam/openrag) ![GitHub Repo stars](https://img.shields.io/github/stars/ShayekhBinIslam/openrag?style=social)
- (EMNLP 2024) **TRACE the evidence: Constructing knowledge-grounded reasoning chains for retrieval-augmented generation** [[Paper]](https://arxiv.org/abs/2406.11460) [[Code]](https://github.com/jyfang6/trace) ![GitHub Repo stars](https://img.shields.io/github/stars/jyfang6/trace?style=social)

## RAG-Enhanced Reasoning
### External Knowledge Retrieval
#### Knowledge Base
- (ICLR 2025) **KBLaM: Knowledge Base augmented Language Model** [[Paper]](https://arxiv.org/pdf/2410.10450) [[Code]](https://github.com/microsoft/KBLaM/) ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/KBLaM?style=social)
- (ArXiv 2025) **Assisting Mathematical Formalization with A Learning-based Premise Retriever** [[Paper]](https://arxiv.org/pdf/2501.13959) [[Code]](https://github.com/ruc-ai4math/Premise-Retrieval) ![GitHub Repo stars](https://img.shields.io/github/stars/ruc-ai4math/Premise-Retrieval?style=social)
- (ArXiv 2025) **ReaRAG: Knowledge-guided Reasoning Enhances Factuality of Large Reasoning Models with Iterative Retrieval Augmented Generation** [[Paper]](https://arxiv.org/pdf/2503.21729) [[Code]](https://github.com/THU-KEG/ReaRAG) ![GitHub Repo stars](https://img.shields.io/github/stars/THU-KEG/ReaRAG?style=social)
- (ArXiv 2025) **Scaling Test-Time Inference with Policy-Optimized, Dynamic Retrieval-Augmented Generation via KV Caching and Decoding** [[Paper]](https://arxiv.org/pdf/2504.01281?)
- (ArXiv 2025) **PIKE-RAG: sPecIalized KnowledgE and Rationale Augmented Generation** [[Paper]](https://arxiv.org/pdf/2501.11551) [[Code]](https://github.com/microsoft/PIKE-RAG?tab=readme-ov-file) ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/PIKE-RAG?style=social)

- (SIGIR 2024) **Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering** [[Paper]](https://dl.acm.org/doi/abs/10.1145/3626772.3661370)
- (ICCBR 2024) **CBR-RAG: Case-Based Reasoning for Retrieval Augmented Generation in LLMs for Legal Question Answering** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-63646-2_29) [[Code]](https://github.com/rgu-iit-bt/cbr-for-legal-rag) ![GitHub Repo stars](https://img.shields.io/github/stars/rgu-iit-bt/cbr-for-legal-rag?style=social)
- (LLM4Code 2024) **LLM-based and Retrieval-Augmented Control Code Generation** [[Paper]](https://dl.acm.org/doi/abs/10.1145/3643795.3648384)
- (ArXiv 2024) **MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries** [[Paper]](https://arxiv.org/pdf/2401.15391) [[Code]](https://github.com/yixuantt/MultiHop-RAG/) ![GitHub Repo stars](https://img.shields.io/github/stars/yixuantt/MultiHop-RAG?style=social)
- (MDPI 2024) **CRP-RAG: A Retrieval-Augmented Generation Framework for Supporting Complex Logical Reasoning and Knowledge Planning** [[Paper]](https://www.mdpi.com/2079-9292/14/1/47)

#### Web Retrieval

- (ICTIR 2025) **Distillation and Refinement of Reasoning in Small Language Models for Document Re-ranking** [[Paper]](https://arxiv.org/pdf/2504.03947) [[Code]](https://github.com/algoprog/InteRank) ![GitHub Repo stars](https://img.shields.io/github/stars/algoprog/InteRank?style=social)
- (NAACL 2025) **Step-by-Step Fact Verification System for Medical Claims with Explainable Reasoning** [[Paper]](https://aclanthology.org/2025.naacl-short.68.pdf) [[Code]](https://github.com/jvladika/StepByStepFV) ![GitHub Repo stars](https://img.shields.io/github/stars/jvladika/StepByStepFV?style=social)

- (COLM 2024) **Web Retrieval Agents for Evidence-Based Misinformation Detection** [[Paper]](https://openreview.net/pdf?id=pKMxO0wBYZ) [[Code]](https://github.com/ComplexData-MILA/webretrieval) ![GitHub Repo stars](https://img.shields.io/github/stars/ComplexData-MILA/webretrieval?style=social)
- (EMNLP 2024) **OPEN-RAG: Enhanced Retrieval-Augmented Reasoning with Open-Source Large Language Models** [[Paper]](https://aclanthology.org/2024.findings-emnlp.831.pdf) [[Code]](https://github.com/ShayekhBinIslam/openrag?tab=readme-ov-file) ![GitHub Repo stars](https://img.shields.io/github/stars/ShayekhBinIslam/openrag?style=social)
- (ACL 2024) **FRVA: Fact-Retrieval and Verification Augmented Entailment Tree Generation for Explainable Question Answering** [[Paper]](https://aclanthology.org/2024.findings-acl.540.pdf)
- (FEVER 2024) **Ragar, your falsehood radar: Rag-augmented reasoning for political fact-checking using multimodal large language models** [[Paper]](https://arxiv.org/pdf/2404.12065)
- (LREC-COLING 2024) **PACAR: Automated Fact-Checking with Planning and Customized Action Reasoning using Large Language Models** [[Paper]](https://aclanthology.org/2024.lrec-main.1099.pdf)

#### Tool Using

- (COLING 2025) **Efficient Tool Use with Chain-of-Abstraction Reasoning** [[Paper]](https://aclanthology.org/2025.coling-main.185.pdf)
- (NAACL 2025) **Meta-Reasoning Improves Tool Use in Large Language Models** [[Paper]](https://arxiv.org/pdf/2411.04535) [[Code]](https://github.com/lisaalaz/tecton?tab=readme-ov-file) ![GitHub Repo stars](https://img.shields.io/github/stars/lisaalaz/tecton?style=social)
- (ArXiv 2025) **Self-Training Large Language Models for Tool-Use Without Demonstrations** [[Paper]](https://arxiv.org/pdf/2502.05867) [[Code]](https://github.com/neneluo/llm-tool-use) ![GitHub Repo stars](https://img.shields.io/github/stars/neneluo/llm-tool-use?style=social)

- (ICLR 2024) **Large Language Models As Tool Makers** [[Paper]](https://arxiv.org/pdf/2305.17126) [[Code]](https://github.com/ctlllll/LLM-ToolMaker) ![GitHub Repo stars](https://img.shields.io/github/stars/ctlllll/LLM-ToolMaker?style=social)
- (ICLR 2024) **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs** [[Paper]](https://openreview.net/pdf?id=dHng2O0Jjr) [[Code]](https://github.com/OpenBMB/ToolBench) ![GitHub Repo stars](https://img.shields.io/github/stars/OpenBMB/ToolBench?style=social)
- (NeurIPS 2024) **AVATAR: Optimizing LLM Agents for Tool Usage via Contrastive Reasoning** [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/2db8ce969b000fe0b3fb172490c33ce8-Paper-Conference.pdf) [[Code]](https://github.com/zou-group/avatar) ![GitHub Repo stars](https://img.shields.io/github/stars/zou-group/avatar?style=social)
- (EMNLP 2024) **Re-Invoke: Tool Invocation Rewriting for Zero-Shot Tool Retrieval** [[Paper]](https://aclanthology.org/2024.findings-emnlp.270.pdf)
- (EMNLP 2024) **SCIAGENT: Tool-augmented Language Models for Scientific Reasoning** [[Paper]](https://aclanthology.org/2024.emnlp-main.880.pdf)
- (EMNLP 2024) **RAR: Retrieval-augmented retrieval for code generation in low-resource languages** [[Paper]](https://aclanthology.org/2024.emnlp-main.1199.pdf)
- (ACL 2024) **MORE: Multi-mOdal REtrieval Augmented Generative Commonsense Reasoning** [[Paper]](https://aclanthology.org/2024.findings-acl.69.pdf) [[Code]](https://github.com/VickiCui/MORE) ![GitHub Repo stars](https://img.shields.io/github/stars/VickiCui/MORE?style=social)
- (LREC-COLING 2024) **Towards Autonomous Tool Utilization in Language Models: A Unified, Efficient and Scalable Framework** [[Paper]](https://aclanthology.org/2024.lrec-main.1427.pdf)
- (NAACL 2024) **Making Language Models Better Tool Learners with Execution Feedback** [[Paper]](https://aclanthology.org/2024.naacl-long.195.pdf) [[Code]](https://github.com/zjunlp/TRICE) ![GitHub Repo stars](https://img.shields.io/github/stars/zjunlp/TRICE?style=social)

- (NeurIPS 2023) **ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings** [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/8fd1a81c882cd45f64958da6284f4a3f-Paper-Conference.pdf) [[Code]](https://github.com/Ber666/ToolkenGPT) ![GitHub Repo stars](https://img.shields.io/github/stars/Ber666/ToolkenGPT?style=social)


### In-context Retrieval

#### Prior Experience

- (ICLR 2025) **Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning** [[Paper]](https://arxiv.org/pdf/2410.19258?) [[Code]](https://github.com/FYYFU/HeadKV/tree/main) ![GitHub Repo stars](https://img.shields.io/github/stars/FYYFU/HeadKV?style=social)
- (ICLR 2025) **Human-like Episodic Memory for Infinite Context LLMs** [[Paper]](https://openreview.net/pdf?id=BI2int5SAC)
- (IEEE TPAMI 2025) **JARVIS-1: Open-World Multi-Task Agents With Memory-Augmented Multimodal Language Models** [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10778628&casa_token=gxJl_piLwrAAAAAA:UCDJkqg5WT7Hr2LSxZUwt6MDTxTH-FHDhL9Dw8XUFJWpcJJSsEMgC5u4wGhE4DvxATX2hK_0CSM)
- (ArXiv 2025) **Reasoning Under 1 Billion: Memory-Augmented Reinforcement Learning for Large Language Models** [[Paper]](https://arxiv.org/pdf/2504.02273)
- (ArXiv 2025) **Review of Case-Based Reasoning for LLM Agents: Theoretical Foundations, Architectural Components, and Cognitive Integration** [[Paper]](https://arxiv.org/pdf/2504.06943?)

- (NeurIPS 2024) **CoPS: Empowering LLM Agents with Provable Cross-Task Experience Sharing** [[Paper]](https://openreview.net/pdf?id=8DLW1saLEY) [[Code]](https://github.com/uclaml/COPS) ![GitHub Repo stars](https://img.shields.io/github/stars/uclaml/COPS?style=social)
- (CHI EA 2024) **"My agent understands me beter": Integrating Dynamic Human-like Memory Recall and Consolidation in LLM-Based** [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3613905.3650839?casa_token=KFkWdHv3aSkAAAAA:lIuVvR1RFiX7HfJ67vl7FVRb5oOKqE1NDl0DSONyvGryZlt5q4WdL8rjp_24NHfcvf0KPAVWGtrBsg)
- (ArXiv 2024) **Large Language Models Orchestrating Structured Reasoning Achieve Kaggle Grandmaster Level** [[Paper]](https://arxiv.org/pdf/2411.03562)
- (ArXiv 2024) **RAP: Retrieval-Augmented Planning with Contextual Memory for Multimodal LLM Agents** [[Paper]](https://arxiv.org/pdf/2402.03610)

#### Example or Training Data

- (ICLR 2025) **OpenRAG: Optimizing RAG End-to-End viaIn-ContextRetrievalLearning** [[Paper]](https://openreview.net/pdf?id=WX0Y0rBsqo)
- (COLING 2025) **PERC: Plan-As-Query Example Retrieval for Underrepresented Code Generation** [[Paper]](https://aclanthology.org/2025.coling-main.534.pdf)

- (IJCAI 2024) **Recall, Retrieve and Reason: Towards Better In-Context Relation Extraction** [[Paper]](https://www.ijcai.org/proceedings/2024/0704.pdf)
- (NeurIPS 2024) **Mixture of Demonstrations for In-Context Learning** [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/a0da098e0031f58269efdcba40eedf47-Paper-Conference.pdf) [[Code]](https://github.com/SongW-SW/MoD?tab=readme-ov-file) ![GitHub Repo stars](https://img.shields.io/github/stars/SongW-SW/MoD?style=social)
- (EACL 2024) **Learning to Retrieve In-Context Examples for Large Language Models** [[Paper]](https://aclanthology.org/2024.eacl-long.105.pdf) [[Code]](https://github.com/microsoft/LMOps/tree/main/llm_retriever) ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/LMOps?style=social)

- (EMNLP 2023) **UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation** [[Paper]](https://aclanthology.org/2023.emnlp-main.758.pdf) [[Code]](https://github.com/microsoft/LMOps) ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/LMOps?style=social)
- (ArXiv 2023) **Dr.ICL: Demonstration-Retrieved In-context Learning** [[Paper]](https://arxiv.org/pdf/2305.14128)

## Synergized RAG and Reasoning

### Reasoning Workflow

#### Chain-based

- (ICLR 2025) **Long-context llms meet rag: Overcoming challenges for long inputs in rag** [[Paper]](https://openreview.net/forum?id=oU3tpaR8fm)
- (ArXiv 2025) **Chain-of-Retrieval Augmented Generation** [[Paper]](https://arxiv.org/abs/2501.14342) [[Code]](https://github.com/microsoft/LMOps/tree/main/corag) ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/LMOps?style=social)
- (ArXiv 2025) **CoT-RAG: Integrating Chain of Thought and Retrieval-Augmented Generation to Enhance Reasoning in Large Language Models** [[Paper]](https://arxiv.org/abs/2504.13534)
- (ArXiv 2025) **Rankcot: Refining knowledge for retrieval-augmented generation through ranking chain-of-thoughts** [[Paper]](https://arxiv.org/abs/2502.17888) [[Code]](https://github.com/NEUIR/RankCoT) ![GitHub Repo stars](https://img.shields.io/github/stars/NEUIR/RankCoT?style=social)

- (EMNLP 2024) **Retrieving, Rethinking and Revising: The Chain-of-Verification Can Improve Retrieval Augmented Generation** [[Paper]](https://arxiv.org/abs/2410.05801)
- (EMNLP 2024) **Chain-of-note: Enhancing robustness in retrieval-augmented language models** [[Paper]](https://arxiv.org/abs/2311.09210)
- (COLM 2024) **Raft: Adapting language model to domain specific rag** [[Paper]](https://openreview.net/pdf?id=rzQGHXNReU) [[Code]](https://github.com/ShishirPatil/gorilla) ![GitHub Repo stars](https://img.shields.io/github/stars/ShishirPatil/gorilla?style=social)
- (ArXiv 2024) **Rat: Retrieval augmented thoughts elicit context-aware reasoning in long-horizon generation** [[Paper]](https://arxiv.org/pdf/2403.05313) [[Code]](https://github.com/CraftJarvis/RAT) ![GitHub Repo stars](https://img.shields.io/github/stars/CraftJarvis/RAT?style=social)
- (ArXiv 2024) **TRACE the evidence: Constructing knowledge-grounded reasoning chains for retrieval-augmented generation** [[Paper]](https://arxiv.org/abs/2406.11460) [[Code]](https://github.com/jyfang6/trace) ![GitHub Repo stars](https://img.shields.io/github/stars/jyfang6/trace?style=social)

- (ACL 2023) **Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions** [[Paper]](https://arxiv.org/pdf/2212.10509) [[Code]](https://github.com/stonybrooknlp/ircot) ![GitHub Repo stars](https://img.shields.io/github/stars/stonybrooknlp/ircot?style=social)

#### Tree-based

- (AAAI 2025) **RATT: A Thought Structure for Coherent and Correct LLM Reasoning** [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/34876) [[Code]](https://github.com/jinghanzhang1998/RATT) ![GitHub Repo stars](https://img.shields.io/github/stars/jinghanzhang1998/RATT?style=social)
- (ArXiv 2025) **MCTS-RAG: Enhance Retrieval-Augmented Generation with Monte Carlo Tree Search** [[Paper]](https://arxiv.org/pdf/2503.20757?) [[Code]](https://github.com/yale-nlp/MCTS-RAG) ![GitHub Repo stars](https://img.shields.io/github/stars/yale-nlp/MCTS-RAG?style=social)
- (ArXiv 2025) **Airrag: Activating intrinsic reasoning for retrieval augmented generation via tree-based search** [[Paper]](https://arxiv.org/abs/2501.10053)
- (ArXiv 2025) **Tree-based RAG-Agent Recommendation System: A Case Study in Medical Test Data** [[Paper]](https://arxiv.org/abs/2501.02727)

- (ArXiv 2024) **SeRTS: Self-Rewarding Tree Search for Biomedical Retrieval-Augmented Generation** [[Paper]](https://arxiv.org/abs/2406.11258)
- (ArXiv 2024) **CORAG: A Cost-Constrained Retrieval Optimization System for Retrieval-Augmented Generation** [[Paper]](https://arxiv.org/abs/2411.00744)

- (ACL 2023) **Tree of clarifications: Answering ambiguous questions with retrieval-augmented large language models** [[Paper]](https://aclanthology.org/2023.emnlp-main.63/) [[Code]](https://github.com/gankim/tree-of-clarifications) ![GitHub Repo stars](https://img.shields.io/github/stars/gankim/tree-of-clarifications?style=social)
- (EMNLP 2023) **Grove: a retrieval-augmented complex story generation framework with a forest of evidence** [[Paper]](https://arxiv.org/abs/2310.05388)

#### Graph-based

##### Walk-on-Graph
- (ICLR 2025) **Reasoning of Large Language Models over Knowledge Graphs with Super-Relations** [[Paper]](https://openreview.net/forum?id=rTCJ29pkuA) [[Code]](https://github.com/SongW-SW/REKNOS) ![GitHub Repo stars](https://img.shields.io/github/stars/SongW-SW/REKNOS?style=social)
- (ICLR 2025) **Simple is Effective: The Roles of Graphs and LLMs in Knowledge-Graph-Based RAG** [[Paper]](https://arxiv.org/pdf/2410.20724) [[Code]](https://github.com/Graph-COM/SubgraphRAG) ![GitHub Repo stars](https://img.shields.io/github/stars/Graph-COM/SubgraphRAG?style=social)
- (ICLR 2025) **StructRAG: Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Structurization** [[Paper]](https://arxiv.org/pdf/2410.08815) [[Code]](https://github.com/icip-cas/StructRAG) ![GitHub Repo stars](https://img.shields.io/github/stars/icip-cas/StructRAG?style=social)
- (ArXiv 2025) **From RAG to Memory: Non-Parametric Continual Learning for Large Language Models** [[Paper]](https://arxiv.org/pdf/2502.14802) [[Code]](https://github.com/OSU-NLP-Group/HippoRAG) ![GitHub Repo stars](https://img.shields.io/github/stars/OSU-NLP-Group/HippoRAG?style=social)
- (ArXiv 2025) **From Local to Global: A GraphRAG Approach to Query-Focused Summarization** [[Paper]](https://arxiv.org/pdf/2404.16130) [[Code]](https://github.com/ksachdeva/langchain-graphrag) ![GitHub Repo stars](https://img.shields.io/github/stars/ksachdeva/langchain-graphrag?style=social)

- (NeurIPS 2024) **G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering** [[Paper]](https://arxiv.org/pdf/2402.07630) [[Code]](https://github.com/manthan2305/Efficient-G-Retriever) ![GitHub Repo stars](https://img.shields.io/github/stars/manthan2305/Efficient-G-Retriever?style=social)
- (NeurIPS 2024) **HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models** [[Paper]](https://arxiv.org/pdf/2405.14831) [[Code]](https://github.com/OSU-NLP-Group/HippoRAG) ![GitHub Repo stars](https://img.shields.io/github/stars/OSU-NLP-Group/HippoRAG?style=social)
- (ArXiv 2024) **DALK: Dynamic Co-Augmentation of LLMs and KG to answer Alzheimer's Disease Questions with Scientific Literature** [[Paper]](https://arxiv.org/pdf/2405.04819) [[Code]](https://github.com/David-Li0406/DALK) ![GitHub Repo stars](https://img.shields.io/github/stars/David-Li0406/DALK?style=social)
- (ArXiv 2024) **GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning** [[Paper]](https://arxiv.org/pdf/2405.20139) [[Code]](https://github.com/cmavro/GNN-RAG) ![GitHub Repo stars](https://img.shields.io/github/stars/cmavro/GNN-RAG?style=social)
- (ArXiv 2024) **LightRAG: Simple and Fast Retrieval-Augmented Generation** [[Paper]](https://arxiv.org/pdf/2410.05779) [[Code]](https://github.com/HKUDS/LightRAG) ![GitHub Repo stars](https://img.shields.io/github/stars/HKUDS/LightRAG?style=social)

- (ArXiv 2023) **Retrieve-Rewrite-Answer: A KG-to-Text Enhanced LLMs Framework for Knowledge Graph Question Answering** [[Paper]](https://arxiv.org/abs/2309.11206) [[Code]](https://github.com/wuyike2000/Retrieve-Rewrite-Answer) ![GitHub Repo stars](https://img.shields.io/github/stars/wuyike2000/Retrieve-Rewrite-Answer?style=social)

- (ICLR 2022) **GreaseLM: Graph REASoning Enhanced Language Models for Question Answering** [[Paper]](https://arxiv.org/pdf/2201.08860) [[Code]](https://github.com/snap-stanford/GreaseLM) ![GitHub Repo stars](https://img.shields.io/github/stars/snap-stanford/GreaseLM?style=social)
- (ACL 2022) **Subgraph Retrieval Enhanced Model for Multi-hop KBQA** [[Paper]](https://arxiv.org/pdf/2202.13296) [[Code]](https://github.com/RUCKBReasoning/SubgraphRetrievalKBQA) ![GitHub Repo stars](https://img.shields.io/github/stars/RUCKBReasoning/SubgraphRetrievalKBQA?style=social)

- (NAACL 2021) **QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering** [[Paper]](https://arxiv.org/pdf/2104.06378) [[Code]](https://github.com/michiyasunaga/qagnn) ![GitHub Repo stars](https://img.shields.io/github/stars/michiyasunaga/qagnn?style=social)

- (ACL 2019) **PullNet: Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text** [[Paper]](https://arxiv.org/pdf/1904.09537) [[Code]](https://github.com/BrambleXu/knowledge-graph-learning/issues/255)

##### Think-on-Graph
- (ICLR 2025) **Think-on-Graph 2.0: Deep and Faithful Large Language Model Reasoning with Knowledge-guided Retrieval Augmented Generation** [[Paper]](https://arxiv.org/pdf/2407.10805) [[Code]](https://github.com/IDEA-FinAI/ToG-2) ![GitHub Repo stars](https://img.shields.io/github/stars/IDEA-FinAI/ToG-2?style=social)

- (ICLR 2024) **Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph** [[Paper]](https://arxiv.org/pdf/2307.07697) [[Code]](https://github.com/spcl/graph-of-thoughts) ![GitHub Repo stars](https://img.shields.io/github/stars/spcl/graph-of-thoughts?style=social)
- (ICLR 2024) **Reasoning on Graphs: Faithful and Interpretable LLM Reasoning (RoG)** [[Paper]](https://arxiv.org/pdf/2310.01061)
- (ACL 2024) **Graph Chain-of-Thought: Augmenting Large Language Models by Reasoning on Graphs** [[Paper]](https://arxiv.org/pdf/2404.07103) [[Code]](https://github.com/PeterGriffinJin/Graph-CoT) ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/Graph-CoT?style=social)
- (ACL 2024) **GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models** [[Paper]](https://arxiv.org/pdf/2406.14550)
- (AAAI 2024) **Knowledge Graph Prompting for Multi-Document Question Answering** [[Paper]](https://arxiv.org/pdf/2308.11730) [[Code]](https://github.com/YuWVandy/KG-LLM-MDQA) ![GitHub Repo stars](https://img.shields.io/github/stars/YuWVandy/KG-LLM-MDQA?style=social)
- (WWW 2025) **Kag: Boosting llms in professional domains via knowledge augmented generation** [[Paper]](https://arxiv.org/abs/2409.13731) [[Code]](https://github.com/OpenSPG/KAG) ![GitHub Repo stars](https://img.shields.io/github/stars/OpenSPG/KAG?style=social)
- (EMNLP 2022) **Empowering Language Models with Knowledge Graph Reasoning for Question Answering** [[Paper]](https://arxiv.org/pdf/2211.08380)
- (CIS 2024) **KnowledgeNavigator: Leveraging Large Language Models for Enhanced Reasoning over Knowledge Graph** [[Paper]](https://arxiv.org/pdf/2312.15880) [[Code]](https://github.com/katzurik/Knowledge_Navigator) ![GitHub Repo stars](https://img.shields.io/github/stars/katzurik/Knowledge_Navigator?style=social)
- (ArXiv 2024) **HyKGE: A Hypothesis Knowledge Graph Enhanced Framework for Accurate and Reliable Medical LLMs Responses** [[Paper]](https://arxiv.org/pdf/2312.15883) [[Code]](https://github.com/Artessay/HyKGE) ![GitHub Repo stars](https://img.shields.io/github/stars/Artessay/HyKGE?style=social)
- (ArXiv 2024) **KG-RAG: Bridging the Gap Between Knowledge and Creativity​** [[Paper]](https://arxiv.org/pdf/2405.12035)
- (ArXiv 2024) **Mitigating Hallucinations in Large Language Models via Self-Refinement-Enhanced Knowledge Retrieval** [[Paper]](https://arxiv.org/pdf/2405.06545)

### Agentic Orchestration

#### Single-Agent

##### Prompting
- (ArXiv 2025) **Search-o1: Agentic Search-Enhanced Large Reasoning Models** [[Paper]](https://arxiv.org/pdf/2501.05366) [[Code]](https://github.com/sunnynexus/Search-o1) ![GitHub Repo stars](https://img.shields.io/github/stars/sunnynexus/Search-o1?style=social)
- (ArXiv 2025) **Plan∗RAG: Efficient Test-Time Planning for Retrieval Augmented Generation** [[Paper]](https://arxiv.org/pdf/2410.20753)
- (ArXiv 2025) **Open Deep Search: Democratizing Search with Open-source Reasoning Agents** [[Paper]](https://arxiv.org/pdf/2503.20201) [[Code]](https://github.com/sentient-agi/OpenDeepSearch) ![GitHub Repo stars](https://img.shields.io/github/stars/sentient-agi/OpenDeepSearch?style=social)
- (ArXiv 2025) **DeepRAG: Thinking to Retrieval Step by Step for Large Language Models** [[Paper]](https://arxiv.org/pdf/2502.01142)
- (ArXiv 2025) **Enhancing Retrieval Systems with Inference-Time Logical Reasoning** [[Paper]](https://arxiv.org/pdf/2503.17860)
- (ArXiv 2025) **Self-Taught Agentic Long-Context Understanding** [[Paper]](https://arxiv.org/pdf/2502.15920)

- (ICLR 2024) **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection** [[Paper]](https://openreview.net/forum?id=hSyW5go0v8) [[Code]](https://github.com/AkariAsai/self-rag) ![GitHub Repo stars](https://img.shields.io/github/stars/AkariAsai/self-rag?style=social)
- (KDD Cup 2024) **A Hybrid RAG System with Comprehensive Enhancement on Complex Reasoning** [[Paper]](https://arxiv.org/pdf/2408.05141)

- (ICLR 2023) **ReAct: Synergizing Reasoning and Acting in Language Models** [[Paper]](https://arxiv.org/pdf/2210.03629) [[Code]](https://github.com/ysymyth/ReAct) ![GitHub Repo stars](https://img.shields.io/github/stars/ysymyth/ReAct?style=social)
- (EMNLP 2023) **Measuring and Narrowing the Compositionality Gap in Language Models** [[Paper]](https://arxiv.org/pdf/2501.05366)
- (ACL 2023) **Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions** [[Paper]](https://arxiv.org/pdf/2410.20753)

##### Supervised Fine-Tuning

- (EMNLP 2024) **REAR: A Relevance-Aware Retrieval-Augmented Framework for Open-Domain Question Answering** [[Paper]](https://arxiv.org/pdf/2402.17497)
- (EMNLP 2024) **RAG-Studio: Towards In-Domain Adaptation of Retrieval Augmented Generation Through Self-Alignment** [[Paper]](https://aclanthology.org/2024.findings-emnlp.41.pdf)
- (ICML 2024) **InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining** [[Paper]](https://arxiv.org/pdf/2310.07713)
- (ICML 2024) **INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning** [[Paper]](https://arxiv.org/pdf/2401.06532) [[Code]](https://github.com/DaoD/INTERS) ![GitHub Repo stars](https://img.shields.io/github/stars/DaoD/INTERS?style=social)
- (ICLR 2024) **Ra-dit: Retrieval-augmented dual instruction tuning** [[Paper]](https://openreview.net/pdf?id=22OTbutug9)
- (COLM 2024) **RAFT: Adapting Language Model to Domain Specific RAG** [[Paper]](https://openreview.net/pdf?id=rzQGHXNReU)
- (SR 2024) **A fine-tuning enhanced RAG system with quantized influence measure as AI judge** [[Paper]](https://www.nature.com/articles/s41598-024-79110-x?utm_source=chatgpt.com)
- (ArXiv 2024) **SFR-RAG: Towards Contextually Faithful LLMs** [[Paper]](https://arxiv.org/pdf/2409.09916) [[Code]](https://github.com/SalesforceAIResearch/SFR-RAG) ![GitHub Repo stars](https://img.shields.io/github/stars/SalesforceAIResearch/SFR-RAG?style=social)

- (NeurIPS 2023) **Toolformer: Language Models Can Teach Themselves to Use Tools** [[Paper]](https://arxiv.org/pdf/2302.04761)

##### Reinforcement Learning

- (ArXiv 2025) **DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments** [[Paper]](https://arxiv.org/pdf/2504.03160?) [[Code]](https://github.com/GAIR-NLP/DeepResearcher) ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher?style=social)
- (ArXiv 2025) **Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning** [[Paper]](https://arxiv.org/pdf/2503.09516) [[Code]](https://github.com/PeterGriffinJin/Search-R1) ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/Search-R1?style=social)
- (ArXiv 2025) **R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning** [[Paper]](https://arxiv.org/pdf/2503.05592) [[Code]](https://github.com/RUCAIBox/R1-Searcher) ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/R1-Searcher?style=social)
- (ArXiv 2025) **ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning** [[Paper]](https://arxiv.org/pdf/2503.19470) [[Code]](https://github.com/Agent-RL/ReSearch) ![GitHub Repo stars](https://img.shields.io/github/stars/Agent-RL/ReSearch?style=social)
- (ArXiv 2025) **ZeroSearch: Incentivize the Search Capability of LLMs without Searching** [[Paper]](https://arxiv.org/pdf/2505.04588) [[Code]](https://github.com/Alibaba-NLP/ZeroSearch) ![GitHub Repo stars](https://img.shields.io/github/stars/Alibaba-NLP/ZeroSearch?style=social)
- (ArXiv 2025) **ReARTeR: Retrieval-Augmented Reasoning with Trustworthy Process Rewarding** [[Paper]](https://arxiv.org/pdf/2501.07861)

- (ACL 2024) **M-RAG: Reinforcing Large Language Model Performance through Retrieval-Augmented Generation with Multiple Partitions**

- (ArXiv 2022) **WebGPT: Browser-assisted question-answering with human feedback**

#### Multi-Agent

- (ACL 2025) **Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research** [[Paper]](https://arxiv.org/pdf/2502.04644) [[Code]](https://github.com/theworldofagents/Agentic-Reasoning) ![GitHub Repo stars](https://img.shields.io/github/stars/theworldofagents/Agentic-Reasoning?style=social)
- (ArXiv 2025) **Collab-RAG: Boosting Retrieval-Augmented Generation for Complex Question Answering via White-Box and Black-Box LLM Collaboration** [[Paper]](https://arxiv.org/pdf/2504.04915)
- (ArXiv 2025) **M-RAG: Reinforcing Large Language Model Performance through Retrieval-Augmented Generation with Multiple Partitions** [[Paper]](https://arxiv.org/pdf/2405.16420)
- (ArXiv 2025) **Knowledge-Aware Iterative Retrieval for Multi-Agent Systems** [[Paper]](https://arxiv.org/pdf/2503.13275)
- (ArXiv 2025) **SLA Management in Reconfigurable Multi-Agent RAG: A Systems Approach to Question Answering** [[Paper]](https://arxiv.org/pdf/2503.10265?)
- (ArXiv 2025) **SurgRAW: Multi-Agent Workflow with Chain of Thought Reasoning for Surgical Intelligence** [[Paper]](https://arxiv.org/pdf/2503.10265?)
- (ArXiv 2025) **HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation** [[Paper]](https://arxiv.org/pdf/2504.12330) [[Code]](https://github.com/ocean-luna/HMRAG) ![GitHub Repo stars](https://img.shields.io/github/stars/ocean-luna/HMRAG?style=social)
- (ArXiv 2025) **RAG-KG-IL: A Multi-Agent Hybrid Framework for Reducing Hallucinations and Enhancing LLM Reasoning through RAG and Incremental Knowledge Graph Learning Integration** [[Paper]](https://arxiv.org/pdf/2503.13514?)
- (ArXiv 2025) **MDocAgent: A Multi-Modal Multi-Agent Framework for Document Understanding** [[Paper]](https://arxiv.org/pdf/2503.13964?) [[Code]](https://github.com/aiming-lab/MDocAgent) ![GitHub Repo stars](https://img.shields.io/github/stars/aiming-lab/MDocAgent?style=social)
- (ArXiv 2025) **MANTRA: Enhancing Automated Method-Level Refactoring with Contextual RAG and Multi-Agent LLM Collaboration** [[Paper]](https://arxiv.org/pdf/2503.14340)
- (ArXiv 2025) **Talk to Right Specialists: Routing and Planning in Multi-agent System for Question Answering** [[Paper]](https://arxiv.org/pdf/2501.07813)
- (ArXiv 2025) **Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning** [[Paper]](https://arxiv.org/pdf/2501.15228)
- (ArXiv 2025) **Agentic Information Retrieval** [[Paper]](https://arxiv.org/abs/2410.09713)

- (NeurIPS 2024) **Chain of Agents: Large Language Models Collaborating on Long-Context Tasks** [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/ee71a4b14ec26710b39ee6be113d7750-Paper-Conference.pdf) [[Code]](https://github.com/richardwhiteii/chain-of-agents) ![GitHub Repo stars](https://img.shields.io/github/stars/richardwhiteii/chain-of-agents?style=social)
- (ArXiv 2024) **A Collaborative Multi-Agent Approach to Retrieval-Augmented Generation Across Diverse Data** [[Paper]](https://arxiv.org/pdf/2412.05838?)
- (ArXiv 2024) **MindSearch: Mimicking Human Minds Elicits Deep AI Searcher** [[Paper]](https://arxiv.org/abs/2407.20183) [[Code]](https://github.com/InternLM/MindSearch) ![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/MindSearch?style=social)

---

*Contributions are welcome! Please feel free to submit pull requests or open issues to suggest new resources.*