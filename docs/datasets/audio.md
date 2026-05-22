# Medical Audio Datasets

This page lists datasets and benchmarks for **medical audio multimodal learning**, including auscultation sounds, cough and respiratory audio, clinical speech, medical dictation, multilingual speech translation, and audio question answering.

Medical audio remains comparatively underrepresented in biomedical MLLM research because clinically useful recordings are privacy-sensitive, device-dependent, noisy, and expensive to annotate. Recent benchmarks increasingly pair audio with natural-language instructions, clinical metadata, reports, or question-answer labels to evaluate whether audio-language models can reason beyond narrow classification.

---

## Audio-Language and Speech Benchmarks

| Dataset | Modalities | Scale | Access | Paper | Date | Tasks |
|---|---|---|---|---|---|---|
| [MedMosaic](https://arxiv.org/abs/2605.00969) | medical audio, text QA | 46.7k QA pairs | paper / pending release | [arXiv](https://arxiv.org/abs/2605.00969) | 2026-05-01 | audio QA, multi-hop reasoning, clinical conversation QA |
| [StethoBench](https://huggingface.co/datasets/hamedfrogh/StethoBench) | heart sounds, lung sounds, instructions, responses | 77.0k instruction-response pairs | public metadata; source audio licensing varies | [arXiv](https://arxiv.org/abs/2603.00355); [HF](https://huggingface.co/datasets/hamedfrogh/StethoBench) | 2026-02-27 | auscultation classification, detection, reporting, reasoning, differential diagnosis |
| CaReSound | cardiac audio, respiratory audio, metadata, QA | annotated benchmark | paper / pending release | [arXiv](https://arxiv.org/abs/2505.01199) | 2025-05-02 | open-ended diagnostic reasoning, heart/lung sound QA |
| [MultiMed-ST](https://github.com/leduckhai/MultiMed-ST) | medical speech, transcripts, translations | 290k samples; Vietnamese, English, German, French, Chinese variants | public code/data/models | [arXiv](https://arxiv.org/abs/2504.03546) | 2025-04-04 | multilingual medical speech translation, ASR, MT |
| MedPodGPT corpus | medical podcast audio, transcripts, retrieval corpus | 4.3k+ audio hours; 39M+ text tokens | preprint / derived corpus | [medRxiv](https://www.medrxiv.org/content/10.1101/2024.07.11.24310304v1) | 2024-07-12 | medical education QA, multilingual transfer, RAG |

---

## Clinical Audio Source Datasets

| Dataset | Modalities | Scale | Access | Paper | Date | Tasks |
|---|---|---|---|---|---|---|
| [CirCor DigiScope](https://physionet.org/content/circor-heart-sound/1.0.3/) | phonocardiogram audio, murmur labels, patient metadata | 5.3k recordings; 1.6k patients | PhysioNet credentialed/open terms | [arXiv](https://arxiv.org/abs/2108.00813); [PhysioNet](https://physionet.org/content/circor-heart-sound/1.0.3/) | 2022-05-30 | murmur detection, heart sound segmentation, auscultation modeling |
| [COUGHVID](https://www.nature.com/articles/s41597-021-00937-4) | cough audio, demographics, COVID status metadata | 20k+ cough recordings | public research dataset | [Scientific Data](https://www.nature.com/articles/s41597-021-00937-4); [arXiv](https://arxiv.org/abs/2009.11644) | 2021-06-23 | cough detection, respiratory screening, acoustic biomarkers |
| [Coswara](https://github.com/iiscleap/Coswara-Data) | breathing, cough, speech, symptoms | 2.7k+ participants | public research dataset | [arXiv](https://arxiv.org/abs/2305.12741); [HF](https://huggingface.co/datasets/szzs1693/coswara-data) | 2023-05-21 | COVID screening, respiratory acoustics, speech/cough analysis |
| [ICBHI 2017 Respiratory Sound Database](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge) | lung sounds, respiratory cycle labels | 920 recordings; 6.9k respiratory cycles; 126 subjects | challenge dataset | [ICBHI Challenge](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge) | 2017 | crackle/wheeze detection, lung sound classification |

---

Back to [README](../../README.md)
