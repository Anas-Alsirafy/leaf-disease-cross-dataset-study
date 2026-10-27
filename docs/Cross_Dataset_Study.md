# Cross-Dataset Study — Five Datasets (Paper 2)

Oct 26, 2026 · Anas Alsirafy

On five public leaf-disease datasets scored on identical leakage-controlled folds, frozen features plus logistic regression match a fine-tuned ConvNeXt-Base on lab images at a fraction of the cost, but fall 16–17 points short on field images. This is the work behind Paper 2, "Cheap Features or Deep Fine-Tuning? Leakage-Controlled Leaf-Disease Classification in the Lab and in the Field" (Kamal Mohamed, Anas Alsirafy, Maki K. Habib). The full IEEE draft exists; a few placeholders remain.

Related sets: cassava, where Pipeline B was developed, starts at [Cassava 1 — Guide](https://github.com/Anas-Alsirafy/cassava-leaf-disease-classification/blob/main/docs/Cassava_1_Guide.md); maize starts at [Cereal 1 — Guide](https://github.com/Anas-Alsirafy/maize-multilabel-foliar-disease/blob/main/docs/Cereal_1_Guide.md).

## The five datasets

| Dataset | Domain | Classes | Images | Exact / near dup. | Cross-label pairs | Groups |
| --- | --- | --- | --- | --- | --- | --- |
| Rice (UCI) | Lab | 3 | 120 | 0 / 1 | 0 | 119 |
| Apple (PlantVillage, color) | Lab | 4 | 3,171 | 7 / 7 | 0 | 3,164 |
| Tomato (PlantVillage, color) | Lab | 10 | 18,160 | 14 / 26 | 0 | 18,136 |
| Cassava (Kaggle 2020) | Field | 5 | 21,397 | 0 / 0 | 0 | 21,397 |
| PlantDoc | Field | 27 | 2,570 (GitHub; Kaggle mirror 2,552) | 17 / 65 | 35 | 2,507 |

All are on Kaggle. Rice: UCI doi 10.24432/C5R013. Cassava: [Kaggle competition](https://www.kaggle.com/competitions/cassava-leaf-disease-classification), CMD is 61.5% of images. PlantDoc: official train and test merged and re-split; the 2-image tomato spider-mite class was dropped. Pipeline A used the GitHub release, Pipeline B the Kaggle mirror.

## Protocol

1. **Duplicate screen.** Resize to 224×224; MD5 of the pixel array (exact copies) and 64-bit dHash (near copies, Hamming distance ≤ 5). Union-find merges pairs into groups.
2. **Folds.** StratifiedGroupKFold on the groups: 5 folds, shuffled, seeds 17, 42, 1337. Every image is tested once per seed; no group spans train and test.
3. **Augmentation.** Only datasets of at most 1,000 images (rice), only the training fold: 7 copies per image with random flips, rotation ±20°, zoom 0.9–1.1, brightness ×0.85–1.15. Copies inherit the parent's group. Two code assertions check no group crosses a split and the test fold is never augmented.
4. **Metrics.** Accuracy, balanced accuracy, macro-F1, per-class P/R/F1, confusion matrices. Uncertainty = s.d. of the three seed means. Rice also gets a Wilson 95% interval on pooled OOF accuracy (24 test images per fold: one error = 4.2 points).
5. **Comparison.** Corrected resampled t-test (Nadeau–Bengio) over paired folds.

## The two pipelines

| Setting | Pipeline A — frozen features | Pipeline B — fine-tuned |
| --- | --- | --- |
| Backbone(s) | MobileNetV2 + EfficientNet-B0 + DenseNet121, ImageNet, frozen | ConvNeXt-Base, ImageNet |
| Input | 224×224, 3×3 Gaussian blur | 384×384, CLAHE on L channel |
| Features / training | 3,584-d concatenated GAP vector, cached once, standardised per fold | Three phases: head only (0.13 M params) → top 30 layers (28.6 M) → full (87.7 M), mixed precision |
| Classifier | Logistic regression, L2, C = 1 (adopted). Also tried: RBF/linear SVM, random forest (500), gradient/histogram boosting, k-NN (k = 5, cosine), soft-voting ensemble | Dense head |
| Validation | — | Group-aware 10% of training fold for early stopping and LR reduction; class weights from training fold |
| Seeds | 17, 42, 1337 | Three seeds for rice, apple, PlantDoc; seed 17 only for tomato and cassava (~4 GPU-h per fold) |
| Control | MobileNetV2-only features (1,280-d) | ConvNeXt-Base kept frozen in all phases (cassava, PlantDoc, seed 17) |

Hardware stated in the paper: NVIDIA Quadro RTX 6000 (24 GB), TensorFlow 2.10, scikit-learn 1.3. Anas reports training on an RTX 5090 — this must be reconciled before submission.

## Results: A vs B on identical folds

| Dataset | A (%) | B (%) | B − A, 95% CI (points) | p |
| --- | --- | --- | --- | --- |
| Rice | 93.64 ± 0.90 | 79.24 ± 3.01 | −14.4 (−28.2, −0.6) | 0.042 |
| Apple | 99.89 ± 0.02 | 99.87 ± 0.03 | −0.02 (−0.22, 0.18) | 0.83 |
| Tomato | 98.86 ± 0.03 | 99.64 (1 seed) | +0.82 (0.26, 1.38) | 0.015 |
| Cassava | 70.61 ± 0.16 | 86.32 (1 seed) | +15.7 (14.5, 17.0) | < 10⁻⁵ |
| Cassava, B frozen | — | 81.88 (1 seed) | +11.3 (9.7, 12.9) | < 10⁻⁴ |
| PlantDoc | 59.50 ± 0.52 | 76.16 ± 0.31 | +16.7 (15.2, 18.1) | < 10⁻¹² |
| PlantDoc, B frozen | — | 71.90 (1 seed) | +12.8 (10.0, 15.7) | < 10⁻³ |

- **Lab images:** A ties B on apple, trails by 0.8 on tomato, leads by 14.4 on rice. B is unstable on rice (folds 56.5–95.8%): 87.7 M parameters fitted to 88 images.
- **Field images:** B leads by 15.7 and 16.7. The frozen ConvNeXt-Base recovers about three-quarters of that gap; fine-tuning adds the last 4.4 points on both (p = 0.006 and p < 10⁻⁵) at about 1.5× the time.
- **Minority classes:** on cassava B reaches balanced accuracy 76.7% and macro-F1 0.762 vs A's 53.6% and 0.534. A's cassava minority classes sit at F1 0.38–0.50; PlantDoc per-class F1 runs from 0.21 to 0.90.
- **Cost:** B took 59.9 GPU-hours (per fold: rice 3 min, apple 34, PlantDoc 33, tomato 246, cassava 262). A's heads took 38 min for all 75 folds plus 363 s to extract features for all 45,418 images.
- **Heads:** logistic regression was most accurate and among the cheapest everywhere. Boosting and the ensemble used 91% of head-fitting time (98.1 of 107.5 h) and scored 0.3–2.5 points lower. Extra backbones over MobileNetV2 alone add 1.40 (rice), 1.28 (apple), 6.86 (tomato), 0.72 (cassava), 9.16 (PlantDoc) points.

## Protocol and data-quality findings

- Augmenting rice before splitting gave a fake 100%; the protocol gives 93.64 ± 0.90% (Wilson 87.8–96.8%). The 6.4-point gap is leakage, larger than any model change.
- PlantDoc has 35 near-identical pairs under different labels, mostly corn gray leaf spot vs blight and potato early vs late blight. In 9, the same file is in the official train and test folders with different labels.

## Against published results

| Dataset | Published (%) | A / B (%) | Reading |
| --- | --- | --- | --- |
| Rice | 73.3–91.3 without enlarging data; 94.4 and 99.4 after augmenting first | 93.6 / 79.2 | A beats every result without augment-first |
| Apple | 98.3–99.6 | 99.9 / 99.9 | Ceiling |
| Tomato | 98.7–99.9 | 98.9 / 99.6 | A in the fine-tuned range with no fine-tuning |
| Cassava | 83.0–89.0; 91.3 Kaggle winner | 70.6 / 86.3 | B in range |
| PlantDoc | 70.5–77.5 | 59.5 / 76.2 | B level with recent models |

A frozen DINOv2-L + CLIP-L fusion reaches 77.0% on PlantDoc (El Karch et al., 2026), suggesting the field-image limit lies in small backbones, not in freezing. Full references: Paper 2, Table IV.

## Preliminary tomato ablation (not leakage-controlled)

MobileNetV2 on 16,011 Kaggle PlantVillage tomato images, one stratified split, seed 42, 3 warm-up + 5 fine-tuning epochs.

| Condition (cumulative) | Acc. (%) | Train (s) |
| --- | --- | --- |
| Frozen MobileNetV2 + dense head | 89.63 | 77 |
| + Gaussian blur | 90.63 | 66 |
| + CLAHE | 89.43 | 65 |
| + standard augmentation | 83.51 | 1,141 |
| + domain-aware augmentation (replaces standard) | 81.93 | 373 |
| + two-phase fine-tuning | 85.55 | 7,286 |

## Known issues and loose ends

- [ ] Hardware line: Quadro RTX 6000 vs RTX 5090 — fix in the paper. Also confirm the TF version (2.10).
- [ ] TBD placeholders: CPU model; Prof. Habib's e-mail; department line and author order; funding statement and grant number.
- [ ] Rerun the tomato ablation under the shared protocol, or keep it labelled preliminary.
- [ ] Pipeline B ran one seed on tomato and cassava: add seeds 42 and 1337 (~80 GPU-hours).
- [ ] The best cassava ensemble (90.32%) was not rerun on the shared folds; see the cassava docs.
- [ ] The frozen control ran one seed on the field sets only, and its head was trained on augmented images rather than cached features.
- [ ] The duplicate screen groups PlantDoc's label-conflicting images but does not correct their labels.
- [x] Say who owns the Pipeline A code (frozen features) and where it lives.

## Future work

- [ ] Clear the loose ends above and agree a venue with Prof. Habib.
- [ ] Add DINOv2 and CLIP features to Pipeline A and rerun cassava and PlantDoc.
- [ ] Produce per-class F1 tables for all five datasets (PlantDoc with a per-fold breakdown).
- [ ] Build one unified protocol table: backbone, input size, batch size, GPU-hours and macro-F1 per dataset.
- [ ] Add maize under the same protocol (see the cereal docs).
