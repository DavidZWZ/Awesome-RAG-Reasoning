# Awesome-RAG-Reasoning

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="Assets/RAG_Reasoning.png" alt="RAG and Reasoning System Overview" width="600"/>
</p>

A curated collection of resources, papers, tools, and implementations that bridge the gap between **Retrieval-Augmented Generation (RAG)** and **Reasoning** in Large Language Models. This repository brings together traditionally separate research domains to enable more powerful AI systems.

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

### Why RAG + Reasoning?
Large Language Models (LLMs) serve as the foundation for modern AI systems, but they face significant limitations in both knowledge access and reasoning capabilities. 
While RAG excels at providing factual knowledge and reasoning excels at logical processing, real-world problems often require both capabilities simultaneously. Complex queries demand not just access to relevant information, but also the ability to reason through that information systematically.

- **Factual + Logical**: RAG provides the factual evidences, reasoning provides the logic Analysis
- **External + Internal**: RAG accesses external knowledge and information, reasoning conducts internal understanding and synthesizes conclusions

**Real-World Impact**: This combination enables AI systems to tackle complex problems that require both knowledge retrieval and sophisticated reasoning, such as scientific research, legal analysis, medical diagnosis, and strategic planning.


### What This Repository Covers

This repository organizes resources across several key areas:

- **📚 Research Papers**: Latest academic publications on RAG and reasoning
- **🔧 Tools & Frameworks**: Open-source implementations and libraries
- **📊 Datasets**: Evaluation benchmarks and training data
- **🎯 Applications**: Real-world use cases and implementations

# 📚 Research Papers

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

### Knowledge Base
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

### Web Retrieval

- (ICTIR 2025) **Distillation and Refinement of Reasoning in Small Language Models for Document Re-ranking** [[Paper]](https://arxiv.org/pdf/2504.03947) [[Code]](https://github.com/algoprog/InteRank) ![GitHub Repo stars](https://img.shields.io/github/stars/algoprog/InteRank?style=social)
- (NAACL 2025) **Step-by-Step Fact Verification System for Medical Claims with Explainable Reasoning** [[Paper]](https://aclanthology.org/2025.naacl-short.68.pdf) [[Code]](https://github.com/jvladika/StepByStepFV) ![GitHub Repo stars](https://img.shields.io/github/stars/jvladika/StepByStepFV?style=social)

- (COLM 2024) **Web Retrieval Agents for Evidence-Based Misinformation Detection** [[Paper]](https://openreview.net/pdf?id=pKMxO0wBYZ) [[Code]](https://github.com/ComplexData-MILA/webretrieval) ![GitHub Repo stars](https://img.shields.io/github/stars/ComplexData-MILA/webretrieval?style=social)
- (EMNLP 2024) **OPEN-RAG: Enhanced Retrieval-Augmented Reasoning with Open-Source Large Language Models** [[Paper]](https://aclanthology.org/2024.findings-emnlp.831.pdf) [[Code]](https://github.com/ShayekhBinIslam/openrag?tab=readme-ov-file) ![GitHub Repo stars](https://img.shields.io/github/stars/ShayekhBinIslam/openrag?style=social)
- (ACL 2024) **FRVA: Fact-Retrieval and Verification Augmented Entailment Tree Generation for Explainable Question Answering** [[Paper]](https://aclanthology.org/2024.findings-acl.540.pdf)
- (FEVER 2024) **Ragar, your falsehood radar: Rag-augmented reasoning for political fact-checking using multimodal large language models** [[Paper]](https://arxiv.org/pdf/2404.12065)
- (LREC-COLING 2024) **PACAR: Automated Fact-Checking with Planning and Customized Action Reasoning using Large Language Models** [[Paper]](https://aclanthology.org/2024.lrec-main.1099.pdf)

### Tool Using

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


## In-context Retrieval

### Prior Experience

- (ICLR 2025) **Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning** [[Paper]](https://arxiv.org/pdf/2410.19258?) [[Code]](https://github.com/FYYFU/HeadKV/tree/main) ![GitHub Repo stars](https://img.shields.io/github/stars/FYYFU/HeadKV?style=social)
- (ICLR 2025) **Human-like Episodic Memory for Infinite Context LLMs** [[Paper]](https://openreview.net/pdf?id=BI2int5SAC)
- (IEEE TPAMI 2025) **JARVIS-1: Open-World Multi-Task Agents With Memory-Augmented Multimodal Language Models** [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10778628&casa_token=gxJl_piLwrAAAAAA:UCDJkqg5WT7Hr2LSxZUwt6MDTxTH-FHDhL9Dw8XUFJWpcJJSsEMgC5u4wGhE4DvxATX2hK_0CSM)
- (ArXiv 2025) **Reasoning Under 1 Billion: Memory-Augmented Reinforcement Learning for Large Language Models** [[Paper]](https://arxiv.org/pdf/2504.02273)
- (ArXiv 2025) **Review of Case-Based Reasoning for LLM Agents: Theoretical Foundations, Architectural Components, and Cognitive Integration** [[Paper]](https://arxiv.org/pdf/2504.06943?)

- (NeurIPS 2024) **CoPS: Empowering LLM Agents with Provable Cross-Task Experience Sharing** [[Paper]](https://openreview.net/pdf?id=8DLW1saLEY) [[Code]](https://github.com/uclaml/COPS) ![GitHub Repo stars](https://img.shields.io/github/stars/uclaml/COPS?style=social)
- (CHI EA 2024) **"My agent understands me beter": Integrating Dynamic Human-like Memory Recall and Consolidation in LLM-Based** [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3613905.3650839?casa_token=KFkWdHv3aSkAAAAA:lIuVvR1RFiX7HfJ67vl7FVRb5oOKqE1NDl0DSONyvGryZlt5q4WdL8rjp_24NHfcvf0KPAVWGtrBsg)
- (ArXiv 2024) **Large Language Models Orchestrating Structured Reasoning Achieve Kaggle Grandmaster Level** [[Paper]](https://arxiv.org/pdf/2411.03562)
- (ArXiv 2024) **RAP: Retrieval-Augmented Planning with Contextual Memory for Multimodal LLM Agents** [[Paper]](https://arxiv.org/pdf/2402.03610)

### Example or Training Data

- (ICLR 2025) **OpenRAG: Optimizing RAG End-to-End viaIn-ContextRetrievalLearning** [[Paper]](https://openreview.net/pdf?id=WX0Y0rBsqo)
- (COLING 2025) **PERC: Plan-As-Query Example Retrieval for Underrepresented Code Generation** [[Paper]](https://aclanthology.org/2025.coling-main.534.pdf)

- (IJCAI 2024) **Recall, Retrieve and Reason: Towards Better In-Context Relation Extraction** [[Paper]](https://www.ijcai.org/proceedings/2024/0704.pdf)
- (NeurIPS 2024) **Mixture of Demonstrations for In-Context Learning** [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/a0da098e0031f58269efdcba40eedf47-Paper-Conference.pdf) [[Code]](https://github.com/SongW-SW/MoD?tab=readme-ov-file) ![GitHub Repo stars](https://img.shields.io/github/stars/SongW-SW/MoD?style=social)
- (EACL 2024) **Learning to Retrieve In-Context Examples for Large Language Models** [[Paper]](https://aclanthology.org/2024.eacl-long.105.pdf) [[Code]](https://github.com/microsoft/LMOps/tree/main/llm_retriever) ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/LMOps?style=social)

- (EMNLP 2023) **UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation** [[Paper]](https://aclanthology.org/2023.emnlp-main.758.pdf) [[Code]](https://github.com/microsoft/LMOps) ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/LMOps?style=social)
- (ArXiv 2023) **Dr.ICL: Demonstration-Retrieved In-context Learning** [[Paper]](https://arxiv.org/pdf/2305.14128)

---

*Contributions are welcome! Please feel free to submit pull requests or open issues to suggest new resources.*