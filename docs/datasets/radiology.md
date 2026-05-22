# Radiology Datasets

This page lists datasets used for training and evaluating multimodal models in **radiology**, including image–text datasets, report datasets, and VQA datasets.

---

## Radiology Multimodal Datasets

| Dataset | Modalities | Images | Text | Paper | Date | Tasks |
|---|---|---|---|---|---|---|
| [Merlin abdominal CT dataset](https://stanfordaimi.azurewebsites.net/datasets/60b9c7ff-877b-48ce-96c3-0194c8205c40) | CT, EHR codes, Text | 25.5k CT scans | 25.5k reports | [Nature](https://www.nature.com/articles/s41586-026-10181-8) | 2026-03-04 | CT VLM pretraining, retrieval, report generation, segmentation, phenotype prediction |
| [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) | 3D CT, Text | 25.7k CT scans | 25.7k reports | [Nature Biomedical Engineering](https://www.nature.com/articles/s41551-025-01599-y); [arXiv](https://arxiv.org/abs/2403.17834) | 2026-02-12 | classification, retrieval, report generation, CT VQA |
| [RadGPT](https://github.com/MrGiovanni/RadGPT) | Img, Text | 2.7M | 1.8M | [arXiv](https://arxiv.org/abs/2501.04678); [ICCV 2025](https://iccv.thecvf.com/virtual/2025/poster/2471) | 2025-10-19 | segmentation, report |
| [RadGenome-Chest CT](https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT) | 3D CT, Mask, Text, VQA | 25.7k CT scans; 197 anatomical mask categories | 665k grounded reports; 1.2M grounded VQA pairs | [Scientific Data](https://www.nature.com/articles/s41597-025-05922-9); [arXiv](https://arxiv.org/abs/2404.16754) | 2025-10-10 | grounded report generation, grounded VQA, region-level CT reasoning |
| [RadMD](https://github.com/chaoyi-wu/RadFM) | Img, Text | 5M | — | [arXiv](https://arxiv.org/abs/2308.02463); [Nat. Commun.](https://www.nature.com/articles/s41467-025-62385-7) | 2025-08-23 | VQA, report generation |
| [PadChest-GR](https://bimcv.cipf.es/bimcv-projects/padchest-gr/) | Img, Bbox, Text | 4.6k | 10.4k | [arXiv](https://arxiv.org/abs/2411.05085); [NEJM AI](https://www.microsoft.com/en-us/research/publication/padchest-gr/?locale=zh-cn) | 2025-06 | grounded report generation |
| [GEMeX](https://www.med-vqa.com/GEMeX/) | Img, Bbox, Text | 151k | 1.6M | [arXiv](https://arxiv.org/abs/2411.16778); [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_GEMeX_A_Large-Scale_Groundable_and_Explainable_Medical_VQA_Benchmark_for_ICCV_2025_paper.pdf) | 2025 | grounded VQA, explainable VQA |
| [ImageCLEF-Med](https://www.imageclef.org/2024/medical/) | Img, Text | 2.8k | 6.4k | [CLEF 2024](https://link.springer.com/chapter/10.1007/978-3-031-71908-0_7) | 2024-09-19 | classification, VQA |
| [ROCOv2](https://github.com/sctg-development/ROCOv2-radiology) | Img, Text | 79k | 79k | [arXiv](https://arxiv.org/abs/2405.10004); [Nat. Sci. Data](https://www.nature.com/articles/s41597-024-03496-6) | 2024-06-26 | classification, captioning |
| [CheXpert Plus](https://aimi.stanford.edu/datasets/chexpert-plus) | Img, Text | 223k | 223k | [arXiv](https://arxiv.org/abs/2405.19538) | 2024-05-29 | classification, captioning, report |
| [CANDID-PTX](https://figshare.com/articles/dataset/CANDID-PTX/14173982) | Img, Mask, Text | 19k | 19k | [Radiology AI](https://pubs.rsna.org/doi/full/10.1148/ryai.2021210136) | 2021-10-13 | classification, segmentation, report |
| [PadChest](https://bimcv.cipf.es/bimcv-projects/padchest/) | Img, Text | 160k | 160k | [arXiv](https://arxiv.org/abs/1901.07441); [Med. Image Anal.](https://www.sciencedirect.com/science/article/abs/pii/S1361841520301614) | 2020-08-20 | classification, captioning |
| [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/) | EHR, Img, Text | 377k | 227k | [arXiv](https://arxiv.org/abs/1901.07042); [Nat. Sci. Data](https://www.nature.com/articles/s41597-019-0322-0) | 2019-12-12 | classification, captioning |
| [VQA-RAD](https://www.nature.com/articles/sdata2018251) | Img, Text | 315 | 3.5k QA pairs | [Scientific Data](https://www.nature.com/articles/sdata2018251); [HF](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) | 2018-11-20 | radiology VQA |
| [ROCO](https://github.com/razorx89/roco-dataset) | Img, Text | 87k | 87k | [LABELS 2018](https://link.springer.com/chapter/10.1007/978-3-030-01364-6_20) | 2018-10-17 | classification, captioning |

---

Back to [README](../../README.md)
