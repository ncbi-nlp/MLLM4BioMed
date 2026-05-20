# Radiology Datasets

This page lists datasets used for training and evaluating multimodal models in **radiology**, including image–text datasets, report datasets, and VQA datasets.

---

## Radiology Multimodal Datasets

| Dataset | Modalities | Images | Text | Paper | Date | Tasks |
|---|---|---|---|---|---|---|
| [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) | Img, Text | 25.6k | 25.6k | [arXiv](https://arxiv.org/abs/2403.17834) | 2024-10-16 | classification, report |
| [RadGPT](https://github.com/MrGiovanni/RadGPT) | Img, Text | 2.7M | 1.8M | [arXiv](https://arxiv.org/abs/2501.04678); ICCV 2025 | 2025-10-19 | segmentation, report |
| [PadChest](https://bimcv.cipf.es/bimcv-projects/padchest/) | Img, Text | 160k | 160k | [arXiv](https://arxiv.org/abs/1901.07441); [Med. Image Anal.](https://www.sciencedirect.com/science/article/abs/pii/S1361841520301614) | 2020-08-20 | classification, captioning |
| [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/) | EHR, Img, Text | 377k | 227k | [arXiv](https://arxiv.org/abs/1901.07042); [Nat. Sci. Data](https://www.nature.com/articles/s41597-019-0322-0) | 2019-12-12 | classification, captioning |
| [CANDID-PTX](https://figshare.com/articles/dataset/CANDID-PTX/14173982) | Img, Mask, Text | 19k | 19k | [Radiology AI](https://pubs.rsna.org/doi/full/10.1148/ryai.2021210136) | 2021-10-13 | classification, segmentation, report |
| [CheXpert Plus](https://aimi.stanford.edu/datasets/chexpert-plus) | Img, Text | 223k | 223k | [arXiv](https://arxiv.org/abs/2405.19538) | 2024-05-29 | classification, captioning, report |
| [PadChest-GR](https://bimcv.cipf.es/bimcv-projects/padchest-gr/) | Img, Bbox, Text | 4.5k | 10.4k | [arXiv](https://arxiv.org/abs/2411.05085) | 2024-11-07 | classification, report |
| [ROCO](https://github.com/razorx89/roco-dataset) | Img, Text | 87k | 87k | [LABELS 2018](https://link.springer.com/chapter/10.1007/978-3-030-01364-6_20) | 2018-10-17 | classification, captioning |
| [ROCOv2](https://github.com/sctg-development/ROCOv2-radiology) | Img, Text | 79k | 79k | [arXiv](https://arxiv.org/abs/2405.10004); [Nat. Sci. Data](https://www.nature.com/articles/s41597-024-03496-6) | 2024-06-26 | classification, captioning |
| [ImageCLEF-Med](https://www.imageclef.org/2024/medical/) | Img, Text | 2.8k | 6.4k | [CLEF 2024](https://link.springer.com/chapter/10.1007/978-3-031-71908-0_7) | 2024-09-19 | classification, VQA |
| [GEMeX](https://www.med-vqa.com/GEMeX/) | Img, Bbox, Text | 151k | 1.6M | [arXiv](https://arxiv.org/abs/2411.16778) | 2024-11-25 | VQA |
| [RadMD](https://github.com/chaoyi-wu/RadFM) | Img, Text | 5M | — | [arXiv](https://arxiv.org/abs/2308.02463); [Nat. Commun.](https://www.nature.com/articles/s41467-025-62385-7) | 2025-08-23 | VQA, report generation |

---

Back to [README](../../READMEv1.md)