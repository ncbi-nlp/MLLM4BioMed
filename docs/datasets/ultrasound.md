# Ultrasound Datasets

This page lists datasets and benchmarks for **ultrasound multimodal learning**, including ultrasound image-text datasets, VQA benchmarks, echocardiography resources, report-generation corpora, and multi-organ ultrasound evaluation sets.

Ultrasound is an important but underrepresented modality for biomedical MLLMs. Compared with CT, MRI, and pathology, ultrasound introduces stronger operator dependence, variable acquisition quality, standard-plane recognition challenges, video dynamics, Doppler/signal interpretation, and specialty-specific reporting language.

---

## Ultrasound Multimodal Datasets and Benchmarks

| Dataset | Modalities | Images / Videos | Text | Paper | Date | Tasks |
|---|---|---|---|---|---|---|
| BUS-CoT | breast ultrasound, annotations, chain-of-thought text | 11.4k breast ultrasound images | CoT rationales, reports, labels | [arXiv](https://arxiv.org/abs/2604.21525) | 2026-04-26 | breast lesion reasoning, report generation, diagnosis |
| Dolphin multimodal ultrasound corpus | ultrasound images, videos, text, QA | multi-source ultrasound corpus | instructions, reports, QA | [arXiv](https://arxiv.org/abs/2509.25748) | 2025-09-30 | ultrasound dialogue, report generation, VQA, classification |
| EchoVLM ultrasound corpus | multi-organ ultrasound images, reports, labels | 208.9k clinical cases; 1.47M images | reports and clinical labels | [arXiv](https://arxiv.org/abs/2509.17174) | 2025-09-18 | universal ultrasound interpretation, classification, report generation |
| ReMUD | ultrasound images, text QA | multi-organ ultrasound benchmark | 45k+ reasoning QA / VQA samples | [arXiv](https://arxiv.org/abs/2506.09131) | 2025-06-10 | ultrasound VQA, diagnostic reasoning, report understanding |
| [U2-BENCH](https://github.com/MAGIC-AI4Med/U2-Bench) | ultrasound images, text QA | 7.2k cases across 15 anatomical regions | task prompts and answers | [arXiv](https://arxiv.org/abs/2505.17779) | 2025-05-23 | ultrasound VQA, plane recognition, diagnosis, measurement, report-related reasoning |
| Ultrasound report generation benchmark | ultrasound images, reports | multi-institution ultrasound images | standardized report text | [arXiv](https://arxiv.org/abs/2505.06812) | 2025-05-11 | ultrasound report generation, standardization |
| [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/) | echocardiography videos, masks | 2D apical 2- and 4-chamber echo sequences from 500 patients | clinical metadata | [IEEE TMI](https://ieeexplore.ieee.org/document/9099400) | 2020-05-25 | cardiac segmentation, view analysis, echo quantification |
| [EchoNet-Dynamic](https://echonet.github.io/dynamic/) | echocardiography videos, clinical labels | 10k+ apical-4-chamber echo videos | ejection fraction labels | [Nature](https://www.nature.com/articles/s41586-020-2145-8) | 2020-03-25 | cardiac function assessment, video understanding |
| [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) | breast ultrasound images, masks | 780 images | diagnostic labels | [Data in Brief](https://www.sciencedirect.com/science/article/pii/S2352340919312181) | 2020-02-18 | breast lesion classification, segmentation |

---

Back to [README](../../README.md)
