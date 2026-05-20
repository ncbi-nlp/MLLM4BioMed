# Radiology MLLMs

This page collects multimodal large language models developed for **radiology imaging tasks**, including chest X-ray, CT, and medical report generation.

---

## Radiology Vision-Language Models

| Model | Paper | Venue | Date |
|---|---|---|---|
| [GLoRIA](https://github.com/marshuang80/gloria) | GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition | [ICCV 2021](https://ieeexplore.ieee.org/document/9710099) | 2021-10-10 |
| [MedViLL](https://github.com/SuperSupermoon/MedViLL) | Multi-Modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Training | [arXiv](https://arxiv.org/abs/2105.11333); [JBHI](https://ieeexplore.ieee.org/document/9894658) | 2022-09-19 |
| [RepsNet](https://sites.google.com/view/repsnet) | RepsNet: Combining Vision with Language for Automated Medical Reports | [arXiv](https://arxiv.org/pdf/2209.13171); [MICCAI 2022](https://link.springer.com/chapter/10.1007/978-3-031-16443-9_68) | 2022-10-01 |
| [ConVIRT](https://github.com/yuhaozhang/convirt) | Contrastive Learning of Medical Visual Representations from Paired Images and Text | [arXiv](https://arxiv.org/abs/2010.00747); [MLCH 2022](https://proceedings.mlr.press/v182/zhang22a.html) | 2022-10-02 |
| [MedCLIP](https://github.com/RyanWangZf/MedCLIP) | MedCLIP: Contrastive Learning from Unpaired Medical Images and Text | [arXiv](https://arxiv.org/abs/2210.10163); [EMNLP 2022](https://aclanthology.org/2022.emnlp-main.256.pdf) | 2022-12-07 |
| [ViewXGen](https://github.com/ttumyche/UniXGen) | Vision-Language Generative Model for View-Specific Chest X-ray Generation | [arXiv](https://arxiv.org/abs/2302.12172); [CHIL 2024](https://proceedings.mlr.press/v248/lee24a.html) | 2023-02-23 |
| [RAMM](https://github.com/GanjinZero/RAMM) | Retrieval-augmented Biomedical Visual Question Answering with Multi-modal Pre-training | [arXiv](https://arxiv.org/abs/2303.00534); [ACM MM 2023](https://dl.acm.org/doi/10.1145/3581783.3611830) | 2023-03-01 |
| [X-REM](https://github.com/rajpurkarlab/X-REM) | Multimodal Image-Text Matching Improves Retrieval-based Chest X-Ray Report Generation | [arXiv](https://arxiv.org/abs/2303.17579); [MIDL 2023](https://openreview.net/forum?id=aZ0OuYMSMMZ) | 2023-03-29 |
| [PubMedCLIP](https://github.com/sarahESL/PubMedCLIP) | How Much Does CLIP Benefit Visual Question Answering in the Medical Domain? | [arXiv](https://arxiv.org/abs/2112.13906); [EACL 2023](https://aclanthology.org/2023.findings-eacl.88) | 2023-05-01 |
| [MedKLIP](https://github.com/MediaBrain-SJTU/MedKLIP) | Medical Knowledge Enhanced Language-Image Pre-Training in Radiology | [arXiv](https://arxiv.org/abs/2301.02228); [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_MedKLIP_Medical_Knowledge_Enhanced_Language-Image_Pre-Training_for_X-ray_Diagnosis_ICCV_2023_paper.pdf) | 2023-10-02 |
| [MUMC](https://github.com/pengfeiliHEU/MUMC) | Masked Vision and Language Pre-training with Unimodal and Multimodal Contrastive Losses for Medical VQA | [arXiv](https://arxiv.org/abs/2307.05314); [MICCAI 2023](https://link.springer.com/chapter/10.1007/978-3-031-43907-0_36) | 2023-10-08 |
| [MAIRA-1](https://www.microsoft.com/en-us/research/project/project-maira/) | A specialised large multimodal model for radiology report generation | [arXiv](https://arxiv.org/abs/2311.13668); [BioNLP-ACL 2024](https://aclanthology.org/2024.bionlp-1.50/) | 2023-11-01 |
| Med-MLLM | A medical multimodal large language model for future pandemics | [npj Digital Medicine](https://www.nature.com/articles/s41746-023-00952-2) | 2023-12-02 |
| [CheXagent](https://github.com/Stanford-AIMI/CheXagent) | A Vision-Language Foundation Model to Enhance Efficiency of Chest X-ray Interpretation | [arXiv](https://arxiv.org/abs/2401.12208) | 2024-01-22 |
| [XrayGPT](https://github.com/mbzuai-oryx/XrayGPT) | Chest Radiographs Summarization using Medical Vision-Language Models | [arXiv](https://arxiv.org/abs/2306.07971); [BioNLP-ACL 2024](https://aclanthology.org/2024.bionlp-1.35.pdf) | 2024-08-16 |
| OaD | Organ-aware Diagnosis Framework for Radiology Report Generation | [IEEE TMI](https://ieeexplore.ieee.org/document/10579857) | 2024-07-01 |
| [MCPL](https://github.com/CUHK-AIM-Group/MCPL) | Multi-Modal Collaborative Prompt Learning for Medical Vision-Language Models | [IEEE TMI](https://ieeexplore.ieee.org/document/10570257) | 2024-06-24 |
| ClinicalBLIP | Vision-Language Model for Generating Textual Descriptions From Clinical Images | [JMIR Formative Research](https://formative.jmir.org/2024/1/e32690) | 2024-08-02 |
| [CT2Rep](https://github.com/ibrahimethemhamamci/CT2Rep) | Automated Radiology Report Generation for 3D Medical Imaging | [arXiv](https://arxiv.org/abs/2403.06801); [MICCAI 2024](https://link.springer.com/chapter/10.1007/978-3-031-72390-2_45) | 2024-10-01 |
| [CTChat](https://github.com/ibrahimethemhamamci/CT-CHAT) | Developing Generalist Foundation Models from Multimodal CT Data | [arXiv](https://arxiv.org/abs/2403.17834) | 2024-10-16 |
| [Merlin](https://github.com/StanfordMIMI/Merlin) | A Vision Language Foundation Model for 3D CT | [arXiv](https://arxiv.org/abs/2406.06512) | 2025-06-28 |
| [LLMSeg](https://github.com/tvseg/MM-LLM-RO) | LLM-driven Multimodal Target Volume Contouring in Radiation Oncology | [arXiv](https://arxiv.org/abs/2311.01908); [Nature Communications](https://www.nature.com/articles/s41467-024-53387-y) | 2024-10-24 |
| Flamingo-CXR | Collaboration Between Clinicians and Vision-Language Models in Radiology Report Generation | [arXiv](https://arxiv.org/abs/2311.18260); [Nature Medicine](https://www.nature.com/articles/s41591-024-03302-1) | 2024-11-07 |
| [MAIRA-2](https://www.microsoft.com/en-us/research/project/project-maira/) | Grounded Radiology Report Generation | [arXiv](https://arxiv.org/abs/2406.04449) | 2024-06-06 |
| [MAIRA-Seg](https://www.microsoft.com/en-us/research/project/project-maira/) | Segmentation-Aware Multimodal Large Language Models | [arXiv](https://arxiv.org/abs/2411.11362); [MLHS 2024](https://proceedings.mlr.press/v259/sharma25a.html) | 2024-12-15 |
| [CXR-LLaVA](https://github.com/ECOFRI/CXR_LLaVA) | Multimodal LLM for Interpreting Chest X-rays | [arXiv](https://arxiv.org/abs/2310.18341); [European Radiology](https://link.springer.com/article/10.1007/s00330-024-11339-6) | 2025-01-15 |
| [RoentGen](https://github.com/StanfordMIMI/RoentGen) | Vision–Language Foundation Model for Generation of Chest X-ray Images | [Nature Biomedical Engineering](https://www.nature.com/articles/s41551-024-01246-y) | 2025-04-09 |
| [RaDialog](https://github.com/ChantalMP/RaDialog) | Vision-Language Model for Radiology Report Generation and Conversational Assistance | [arXiv](https://arxiv.org/abs/2311.18681); MIDL 2025 | 2025-07-09 |
| [RadFM](https://github.com/chaoyi-wu/RadFM) | Generalist Radiology Foundation Model Using Web-scale 2D & 3D Data | [arXiv](https://arxiv.org/abs/2308.02463); [Nature Communications](https://www.nature.com/articles/s41467-025-62385-7) | 2025-08-23 |

---

Back to [README](../../READMEv1.md)