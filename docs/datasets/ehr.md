# Electronic Health Record Datasets

This page lists datasets and benchmarks for **electronic health record (EHR)** multimodal modeling, including structured EHR tables, clinical notes, longitudinal ICU/ED time series, imaging, waveforms, genomics, and wearable-linked patient records.

---

## EHR and Longitudinal Multimodal Datasets

| Dataset | Modalities | Scale | Access | Paper | Date | Tasks |
|---|---|---|---|---|---|---|
| [MIMIC-IV-Ext-CLIF](https://physionet.org/content/mimic-iv-ext-clif/1.1.0/) | EHR, longitudinal ICU | MIMIC-IV v3.1 in CLIF; 14 CLIF tables | credentialed | [PhysioNet](https://physionet.org/content/mimic-iv-ext-clif/1.1.0/) | 2026-03-23 | harmonized ICU analytics, multi-center benchmarking |
| [UK Biobank](https://www.ukbiobank.ac.uk/about-our-data/) | EHR, imaging, omics, phenotypes | 500k participants | approved research access | [EHR linkage](https://biobank.ctsu.ox.ac.uk/crystal/crystal/docs/DataLinkageProcess.pdf); [Imaging data](https://community.ukbiobank.ac.uk/hc/en-gb/articles/24618819821981-Imaging-Data) | 2025-12-24 | population health, imaging-genomics, risk prediction |
| [MC-MED](https://physionet.org/content/mc-med/) | EHR, notes, vitals, waveforms | 118.4k ED visits | credentialed | [Scientific Data](https://www.nature.com/articles/s41597-025-05419-5) | 2025-07-01 | ED monitoring, decompensation, disposition, revisit prediction |
| [All of Us Research Program](https://www.researchallofus.org/) | EHR, surveys, physical measures, wearables, genomics | 1M target cohort | tiered Researcher Workbench access | [Data types](https://support.researchallofus.org/hc/en-us/articles/4619151535508-Data-Types-and-Organization); [Data methods](https://www.researchallofus.org/data-tools/methods/) | 2025-04-25 | precision medicine, cohort discovery, risk prediction |
| [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) | EHR, ICU, ED | 364.6k patients; 546k hospitalizations; 94.5k ICU stays | credentialed | [Scientific Data](https://www.nature.com/articles/s41597-022-01899-x) | 2024-10-11 | clinical prediction, phenotyping, cohort discovery |
| [EHRXQA](https://physionet.org/content/ehrxqa/1.0.0/) | EHR tables, CXR images, QA | 46.2k samples; 417 templates | credentialed | [NeurIPS 2023](https://arxiv.org/abs/2310.18652) | 2024-07-23 | multimodal EHR QA, semantic parsing, VQA |
| [MIMIC-Ext-MIMIC-CXR-VQA](https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/) | EHR, CXR images, VQA | 377k QA entries | credentialed | [PhysioNet](https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/) | 2024-07-19 | medical VQA, instruction tuning |
| [MedPromptX-VQA](https://github.com/BioMedIA-MBZUAI/MedPromptX) | EHR, CXR images, VQA | interleaved image-EHR VQA | public code/data release | [arXiv](https://arxiv.org/abs/2403.15585) | 2024-05-12 | in-context VQA, CXR diagnosis |
| [AmsterdamUMCdb](https://github.com/AmsterdamUMC/AmsterdamUMCdb) | ICU EHR, vitals, labs, medications | 23.1k admissions; 20.1k patients | approved research access | [Crit. Care Med.](https://journals.lww.com/ccmjournal/fulltext/2021/06000/sharing_icu_patient_data_responsibly_under_the.16.aspx) | 2024-05 | ICU prediction, external validation |
| [INSPECT](https://aimi.stanford.edu/datasets/inspect-Multimodal-Dataset-for-Pulmonary-Embolism-Diagnosis-and-Prognosis) | CT, EHR, reports | 19.4k patients | non-commercial DUA | [arXiv](https://arxiv.org/abs/2311.10798); [Stanford AIMI](https://aimi.stanford.edu/data) | 2023-11-17 | pulmonary embolism diagnosis and prognosis |
| [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/) | clinical notes, radiology reports | 331.8k discharge summaries; 2.3M radiology reports | credentialed | [PhysioNet](https://physionet.org/content/mimic-iv-note/2.2/) | 2023-01-06 | clinical NLP, summarization, coding |
| [HiRID](https://physionet.org/content/hirid/1.1.1/) | ICU EHR, high-resolution time series | 34k admissions; 681 variables | contributor review | [PhysioNet](https://physionet.org/content/hirid/1.1.1/) | 2021-02-18 | early warning, circulatory failure, ICU forecasting |
| [eICU-CRD](https://physionet.org/content/eicu-crd/2.0/) | ICU EHR, vitals, labs, diagnoses, treatments | 200k+ ICU admissions; 208 hospitals | credentialed | [Scientific Data](https://www.nature.com/articles/sdata2018178) | 2019-04-15 | external validation, ICU prediction |
| [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) | ICU EHR, notes, imaging reports | 40k+ ICU patients | credentialed | [Scientific Data](https://www.nature.com/articles/sdata201635) | 2016-09-04 | clinical prediction, clinical NLP, benchmarking |

---

## EHR Benchmarks and Evaluation Protocols

| Benchmark | Modalities | Scale | Access | Paper | Date | Tasks |
|---|---|---|---|---|---|---|
| [TIMER / TIMER-Eval](https://github.com/som-shahlab/TIMER) | longitudinal EHR, timestamp-linked instructions, clinical text | STARR-derived temporal instruction and evaluation framework | code public; underlying EHR restricted | [npj Digital Medicine](https://www.nature.com/articles/s41746-025-01965-9) | 2025-09-26 | temporal EHR reasoning, longitudinal QA, timeline-grounded instruction tuning |
| MEDFuse benchmark protocol | structured EHR labs, clinical notes | MIMIC-III plus in-house FEMH validation | MIMIC-III credentialed; FEMH not public | [arXiv](https://arxiv.org/abs/2407.12309) | 2024-07-17 | multimodal EHR fusion, 10-disease multi-label classification |

---

Back to [README](../../README.md)
