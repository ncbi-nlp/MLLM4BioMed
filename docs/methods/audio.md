# Medical Audio MLLMs

This page collects multimodal and large-language-model methods for **medical audio**, including clinical speech, medical dictation, auscultation sounds, cough and respiratory audio, audio question answering, and speech-driven clinical image interaction.

Medical audio offers a clinically important but still underdeveloped modality for MLLMs. Compared with images and text, audio systems must handle recording-device variability, background noise, short and sparse signals, privacy constraints, limited labels, and clinical vocabulary. Recent work addresses these gaps through audio encoders connected to LLMs, instruction-tuned auscultation datasets, medical ASR, and multimodal interfaces that combine speech with images or EHR-derived context.

---

## Medical Audio and Speech-Language Models

| Model | Paper | Venue | Date |
|---|---|---|---|
| MedASR | MedASR: An Open-Source Model for High-Accuracy Medical Dictation | [arXiv](https://arxiv.org/abs/2605.16555); [Google Developers](https://developers.google.com/health-ai-developer-foundations/medasr) | 2026-05-15 |
| [Au-M-ol](https://arxiv.org/abs/2604.23284) | Au-M-ol: A Unified Model for Medical Audio and Language Understanding | [arXiv](https://arxiv.org/abs/2604.23284) | 2026-04-25 |
| [StethoLM](https://huggingface.co/askyishan/StethoLM) | StethoLM: Audio Language Model for Cardiopulmonary Analysis Across Clinical Tasks | [arXiv](https://arxiv.org/abs/2603.00355); [TMLR](https://openreview.net/forum?id=i9RuUH9Jyj) | 2026-02-27 |
| MedSpeak | MedSpeak: A Knowledge Graph-Aided ASR Error Correction Framework for Spoken Medical QA | [arXiv](https://arxiv.org/abs/2602.00981) | 2026-02-01 |
| Few-Shot Spectrogram Lung Sound MLLM | Evaluating Few-Shot Prompting for Spectrogram-Based Lung Sound Classification Using a Multimodal Language Model | [medRxiv](https://www.medrxiv.org/content/10.1101/2025.07.27.25332255v1) | 2025-07-27 |
| [CaReAQA](https://arxiv.org/abs/2505.01199) | CaReAQA: A Cardiac and Respiratory Audio Question Answering Model for Open-Ended Diagnostic Reasoning | [arXiv](https://arxiv.org/abs/2505.01199); CHIL 2025 | 2025-05-02 |
| [SilVar-Med](https://arxiv.org/abs/2504.10642) | SilVar-Med: A Speech-Driven Visual Language Model for Explainable Abnormality Detection in Medical Imaging | [arXiv](https://arxiv.org/abs/2504.10642) | 2025-04-14 |
| [MultiMed-ST](https://github.com/leduckhai/MultiMed-ST) | MultiMed-ST: Large-scale Many-to-many Multilingual Medical Speech Translation | [arXiv](https://arxiv.org/abs/2504.03546) | 2025-04-04 |
| MedPodGPT | MedPodGPT: A multilingual audio-augmented large language model for medical research and education | [medRxiv](https://www.medrxiv.org/content/10.1101/2024.07.11.24310304v1) | 2024-07-12 |

---

## Research Themes

| Theme | Representative Work | Notes |
|---|---|---|
| Medical ASR and dictation | MedASR, Au-M-ol, MedSpeak | Targets clinical vocabulary, noisy speech, dictation, and spoken medical QA pipelines. |
| Auscultation reasoning | StethoLM, CaReAQA | Moves heart/lung sound analysis from closed-label classification toward instruction following, reports, and differential diagnosis. |
| Multilingual clinical communication | MultiMed-ST | Evaluates many-to-many speech translation for medical communication across languages. |
| Audio-visual clinical interaction | SilVar-Med | Uses spoken instructions to query medical images and generate explainable abnormality analysis. |
| Prompting and spectrogram interfaces | Few-shot lung sound MLLM study | Tests whether general multimodal LLMs can reason over spectrogram images without task-specific training. |
| Medical education audio | MedPodGPT | Uses medical podcasts and retrieval to improve multilingual medical research and education support. |

---

Back to [README](../../README.md)
