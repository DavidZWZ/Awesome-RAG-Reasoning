# Awesome-RAG-Reasoning

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A curated collection of resources, papers, tools, and implementations that bridge the gap between **Retrieval-Augmented Generation (RAG)** and **Reasoning** in Large Language Models. This repository brings together traditionally separate research domains to enable more powerful AI systems.

## 📖 Introduction

**Retrieval-Augmented Generation (RAG)** has emerged as a powerful paradigm that combines the strengths of large language models with external knowledge retrieval. By augmenting language models with relevant information from external sources, RAG systems can provide more accurate, up-to-date, and factual responses while maintaining the generative capabilities of modern LLMs.

**Reasoning** has recently gained significant popularity as a complementary approach to enhance LLM performance. Reasoning techniques focus on improving the model's ability to process information, perform logical analysis, and arrive at conclusions through structured thinking processes. These methods enable LLMs to tackle complex problems that require multi-step inference, causal understanding, and systematic problem-solving.

### Two Complementary Paradigms for LLM Enhancement

Modern approaches to improving LLM performance have converged on two main paradigms, each addressing different aspects of the model's capabilities. **These domains have been developed largely independently**, with separate research communities, methodologies, and evaluation benchmarks:

**🔍 RAG Community**: Focused on knowledge retrieval, document processing, and factual grounding
- **Core Concept**: Retrieves and incorporates external knowledge to augment the model's factual base
- **Primary Goal**: Bridge the knowledge gap by providing access to information beyond training data
- **Key Mechanism**: Query → Retrieve → Augment → Generate
- **Use Cases**: Question answering, fact verification, domain-specific applications
- **Limitations**:
  - Falls short on problems requiring multi-step thinking
  - May retrieve irrelevant or outdated information
  - Limited by the quality and coverage of external knowledge bases

**🧠 Reasoning Community**: Focused on logical inference, step-by-step thinking, and problem decomposition
- **Core Concept**: Enhances the model's ability to manipulate and reason with internal knowledge
- **Primary Goal**: Improve logical thinking, step-by-step analysis, and problem-solving capabilities
- **Key Mechanism**: Problem → Decompose → Reason → Synthesize
- **Use Cases**: Mathematical reasoning, logical puzzles, strategic planning, causal analysis
- **Limitations**:
  - Often hallucinates or mis-grounds facts
  - May produce logically sound but factually incorrect conclusions
  - Struggles with up-to-date or domain-specific information

**This repository serves as a comprehensive collection that bridges these traditionally separate domains**, providing resources for researchers and practitioners interested in combining the strengths of both approaches.

### Why RAG + Reasoning?
Large Language Models (LLMs) serve as the foundation for modern AI systems, but they face significant limitations in both knowledge access and reasoning capabilities. 
While RAG excels at providing factual knowledge and reasoning excels at logical processing, real-world problems often require both capabilities simultaneously. Complex queries demand not just access to relevant information, but also the ability to reason through that information systematically.

**Different Perspectives**:
- **Factual + Logical**: RAG provides the factual evidences, reasoning provides the logic Analysis
- **External + Internal**: RAG accesses external knowledge and information, reasoning conducts internal understanding and synthesizes conclusions

**Real-World Impact**: This combination enables AI systems to tackle complex problems that require both knowledge retrieval and sophisticated reasoning, such as scientific research, legal analysis, medical diagnosis, and strategic planning.


### What This Repository Covers

This repository organizes resources across several key areas:

- **📚 Research Papers**: Latest academic publications on RAG and reasoning
- **🔧 Tools & Frameworks**: Open-source implementations and libraries
- **📊 Datasets**: Evaluation benchmarks and training data
- **🎯 Applications**: Real-world use cases and implementations







---

*Contributions are welcome! Please feel free to submit pull requests or open issues to suggest new resources.*