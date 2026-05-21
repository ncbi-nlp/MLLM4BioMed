# Histopathology Datasets

This page lists datasets used for training and evaluating multimodal models in **computational histopathology**, including whole-slide imaging datasets, pathology VQA datasets, and multimodal pathology resources.

---

## Histopathology Multimodal Datasets

| Dataset | Modalities | Images | Text | Paper | Date | Tasks |
|---|---|---|---|---|---|---|
| WSI-Bench | Img, Text | 180k | 180k | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Liang_WSI-LLaVA_A_Multimodal_Large_Language_Model_for_Whole_Slide_Image_ICCV_2025_paper.html) | 2025-10-19 | VQA |
| [WSI-VQA](https://github.com/cpystan/WSI-VQA) | Img, Text | 977 | 8.6k | [arXiv](https://arxiv.org/abs/2407.05603); [ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-72764-1_23) | 2024-10-25 | VQA |
| QUILT-Instruct | Img, Text | 107k | 107k | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Seyfioglu_Quilt-LLaVA_Visual_Instruction_Tuning_by_Extracting_Localized_Narratives_from_Open-Source_CVPR_2024_paper.html) | 2024-06-09 | VQA |
| [Quilt-1M](https://quilt1m.github.io/) | Img, Audio, Text | 802k | 802k | [arXiv](https://arxiv.org/abs/2306.11207); [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/73608) | 2023-09-25 | classification, report |
| [OpenPath](https://github.com/PathologyFoundation/plip) | Img, Text | 208k | 208k | [Nat. Med.](https://www.nature.com/articles/s41591-023-02504-3) | 2023-08-17 | classification, report |
| [PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa) | Img, Text | 4.9k | 32.7k | [arXiv](https://arxiv.org/abs/2003.10286) | 2020-05-07 | VQA |
| [TCGA](https://portal.gdc.cancer.gov/) | Img, Gene, Text | 44k* | — | — | — | classification, survival analysis |

---

Back to [README](../../README.md)
