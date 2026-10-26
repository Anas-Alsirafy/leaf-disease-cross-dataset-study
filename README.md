# Cross-Dataset Study (Paper 2) and PlantVillage

Code and reference for the five-dataset comparison behind Paper 2, *"Cheap Features or Deep Fine-Tuning? Leakage-Controlled Leaf-Disease Classification in the Lab and in the Field"* (Kamal Mohamed, Anas Alsirafy, Maki K. Habib — AUC). This folder also holds the PlantVillage notebook.

**Headline finding:** on identical leakage-controlled folds, frozen ImageNet features with logistic regression match a fine-tuned ConvNeXt-Base on lab images at a fraction of the cost, but fall 16–17 points short on field images.

> ⚠️ **Most of the Paper 2 code is not in this repository.** Only the PlantVillage notebook is here. See [What's missing](#whats-missing).

## Repository layout

```
leaf-disease-cross-dataset-study/
├── results.csv   # every reported number, one row per metric
├── plantvillage_apple_tomatopepper_v3.ipynb   # ConvNeXt-Base on PlantVillage apple and tomato+pepper
└── README.md
```

## Results on identical folds

Pipeline A: frozen MobileNetV2 + EfficientNet-B0 + DenseNet121 features + logistic regression. Pipeline B: ConvNeXt-Base fine-tuned in three phases.

| Dataset | Domain | A (%) | B (%) | B − A, 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| Rice (UCI) | Lab | 93.64 ± 0.90 | 79.24 ± 3.01 | −14.4 (−28.2, −0.6) | 0.042 |
| Apple (PlantVillage) | Lab | 99.89 ± 0.02 | 99.87 ± 0.03 | −0.02 (−0.22, 0.18) | 0.83 |
| Tomato (PlantVillage) | Lab | 98.86 ± 0.03 | 99.64 (1 seed) | +0.82 (0.26, 1.38) | 0.015 |
| Cassava (Kaggle 2020) | Field | 70.61 ± 0.16 | 86.32 (1 seed) | +15.7 (14.5, 17.0) | < 1e-5 |
| PlantDoc | Field | 59.50 ± 0.52 | 76.16 ± 0.31 | +16.7 (15.2, 18.1) | < 1e-12 |

± is the s.d. of three seed means. A ConvNeXt-Base kept frozen recovers about three-quarters of the field-image gap; fine-tuning adds the last 4.4 points. Pipeline B cost 59.9 GPU-hours; Pipeline A's classifier heads took 38 minutes for all 75 folds.

## Evaluation protocol

Every number above was produced under this protocol. Reuse it for any new dataset.

1. **Duplicate screen.** Resize to 224×224. Compute MD5 of the pixel array for exact copies and a 64-bit dHash for near copies (Hamming distance ≤ 5). Merge matching pairs into groups with union-find.
2. **Folds.** `StratifiedGroupKFold` on the duplicate groups: 5 folds, shuffled, seeds **17, 42, 1337**. No group spans train and test.
3. **Augmentation.** Only inside the training fold, and only for datasets of at most 1,000 images. Copies inherit their parent's group. Assert that no group crosses a split and that the test fold is never augmented.
4. **Metrics.** Accuracy, balanced accuracy, macro-F1, per-class precision/recall/F1, confusion matrices. Uncertainty is the s.d. of the three seed means.
5. **Comparison.** Corrected resampled t-test (Nadeau & Bengio, 2003) over paired folds.

| Dataset | Images | Exact / near duplicates | Cross-label pairs | Groups |
| --- | --- | --- | --- | --- |
| Rice | 120 | 0 / 1 | 0 | 119 |
| Apple | 3,171 | 7 / 7 | 0 | 3,164 |
| Tomato | 18,160 | 14 / 26 | 0 | 18,136 |
| Cassava | 21,397 | 0 / 0 | 0 | 21,397 |
| PlantDoc | 2,570 | 17 / 65 | 35 | 2,507 |

**PlantDoc warning:** 35 near-identical image pairs carry different labels, and in 9 of them the same file sits in both the official train and test folders. Never report on PlantDoc's official split alone.

## PlantVillage notebook — `plantvillage_apple_tomatopepper_v3.ipynb`

Trains two ConvNeXt-Base classifiers on PlantVillage colour images: **apple** (4 classes) and **tomato + bell pepper** (all classes combined into one model). This is the latest of three versions; v1 and v2 are superseded.

> This notebook uses plain `StratifiedKFold`, not the grouped protocol above, and merges tomato with pepper. It is **not** the Pipeline B run reported in Paper 2, which used tomato alone (10 classes) on the shared grouped folds.

### Setup

```bash
pip install tensorflow scikit-learn opencv-python pandas numpy matplotlib seaborn
```

Download PlantVillage from Kaggle and the ConvNeXt-Base no-top weights (`.h5`). Then set the paths in Cell 2:

```python
CONFIG = {
    'color_dir':               r'C:\path\to\PlantVillage\plantvillage dataset\color',
    'apple_clahe_dir':         r'C:\path\to\clahe\apple',
    'tomato_pepper_clahe_dir': r'C:\path\to\clahe\tomato_pepper',
    'convnext_weights':        r'C:\path\to\convnext_base_notop.h5',
    'save_dir':                r'C:\path\to\save\models',
    ...
}
```

### Cells

| Cell | What it does |
| --- | --- |
| 1–3 | Imports, `CONFIG`, mixed precision |
| 4 | `build_df_from_prefixes()`: picks class folders by crop prefix (e.g. `['Tomato', 'Pepper']`) |
| 5 | CLAHE pre-processing into the `*_clahe_dir` folders |
| 6 | Pre-processing, augmentation, dataset and model functions |
| 7 | `train_crop()`: 5-fold, three-phase training with balanced class weights and early stopping |
| 8 | Train apple → `fold_{k}_apple_convnext.keras`, results JSON, confusion matrix |
| 9 | Train tomato + pepper → `fold_{k}_tomato_pepper_convnext.keras` |
| 10 | Re-evaluate saved models on the same 5-fold splits → `{prefix}_eval_results.json`, `{prefix}_eval_confusion.png` |

### Training settings

| Phase | Trainable | LR | Max epochs | Early-stopping patience |
| --- | --- | --- | --- | --- |
| 1 | Head only | 1e-4 | 25 | 7 |
| 2A | Top 30 backbone layers | 1e-5 | 20 | 7 |
| 2B | Whole network | 5e-6 | 10 | 6 |

`img_size=384`, `batch_size=32`, `num_folds=5`, `seed=42`, CLAHE pre-processing, mixed precision.

## What's missing

These pieces of Paper 2 are not in this repository. Get them from the Pipeline A owner and Anas's Google Drive:

- [ ] Pipeline A: feature extraction, caching and the six classifier heads
- [ ] Pipeline B: the rerun on the shared grouped folds (all five datasets, plus the frozen-backbone control)
- [ ] Duplicate screen (MD5 + dHash + union-find) and fold generation
- [ ] Statistics: corrected resampled t-test, Wilson interval
- [ ] The paper draft (`Paper2_IEEE_Draft_1.docx`)

## Paper 2 loose ends

- [ ] Hardware line says Quadro RTX 6000; training was on an RTX 5090
- [ ] Placeholders: CPU model, Prof. Habib's e-mail, department line/author order, funding statement
- [ ] Add seeds 42 and 1337 to Pipeline B on tomato and cassava (~80 GPU-hours)
- [ ] Rerun the preliminary tomato ablation under the shared protocol

## Related

- [cassava-leaf-disease-classification](https://github.com/Anas-Alsirafy/cassava-leaf-disease-classification) — where Pipeline B's training recipe was developed
- [maize-multilabel-foliar-disease](https://github.com/Anas-Alsirafy/maize-multilabel-foliar-disease) — multi-label maize (cereal) work

## Contact

Anas Alsirafy, American University in Cairo (anas.samy@aucegypt.edu). Supervisor: Prof. Maki K. Habib. Co-author: Kamal Mohamed (kamal_mohamed@aucegypt.edu).
