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

## 📚 Research Papers

### Retrieval Optimization
- (AAAI 2025) **MaFeRw: Query Rewriting with Multi-Aspect Feedbacks for Retrieval-Augmented Large Language Models** [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/34732) [[Code]](https://github.com/TAP-LLM/MaFeRw) ![GitHub Repo stars](https://img.shields.io/github/stars/TAP-LLM/MaFeRw?style=social)
- (ArXiv 2025) **Collab-RAG: Boosting Retrieval-Augmented Generation for Complex Question Answering via White-Box and Black-Box LLM Collaboration** [[Paper]](https://arxiv.org/abs/2504.04915) [[Code]](https://github.com/ritaranx/Collab-RAG/) ![GitHub Repo stars](https://img.shields.io/github/stars/ritaranx/Collab-RAG?style=social)
- (ArXiv 2025) **DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning** [[Paper]](https://arxiv.org/abs/2503.00223) [[Code]](https://github.com/pat-jj/DeepRetrieval) ![GitHub Repo stars](https://img.shields.io/github/stars/pat-jj/DeepRetrieval?style=social)
- (ArXiv 2025) **Credible plan-driven rag method for multi-hop question answering** [[Paper]](https://arxiv.org/abs/2504.16787)
- (ArXiv 2025) **FIND: Fine-grained Information Density Guided Adaptive Retrieval-Augmented Generation for Disease Diagnosis** [[Paper]](https://arxiv.org/abs/2502.14614)
- (ArXiv 2025) **LLM-Independent Adaptive RAG: Let the Question Speak for Itself** [[Paper]](https://arxiv.org/abs/2505.04253) [[Code]](https://github.com/marialysyuk/External_Adaptive_Retrieval) ![GitHub Repo stars](https://img.shields.io/github/stars/marialysyuk/External_Adaptive_Retrieval?style=social)

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








---

*Contributions are welcome! Please feel free to submit pull requests or open issues to suggest new resources.*