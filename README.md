# Amazon ML Challenge 2025 — Multimodal Price Prediction

> **Final Ensemble SMAPE: 41.5%** — Multimodal regression combining text (BERT / SBERT), vision (DINOv2), and engineered numeric features.

## Approach

A multi-input regression model concatenates:
- **Text embeddings** — BERT `pooler_output` or Sentence-BERT (all-mpnet-base-v2) sentence embeddings
- **Image embeddings** — DINOv2 ViT-B14 (self-supervised, 768-d)
- **Numeric features** — `Count`, `oz`, `fl_oz` (engineered from catalog content)

These are fed to an MLP regression head predicting `log1p(price)`. A KNN variant adds neighbor price/feature context via FAISS.

## Project Structure

```
├── code/
│   ├── preprocessing/
│   │   ├── create_splits.py          # 85/15 train-val split
│   │   ├── preprocess_stage1.py      # Extract value/unit from catalog, normalize units
│   │   └── preprocess_stage2.py      # Convert to oz/fl_oz base, extract pack count
│   ├── images/
│   │   ├── download_images.py        # Multithreaded image downloader (wget)
│   │   └── create_dino_embeddings.py # DINOv2 embeddings with DDP support
│   └── training/
│       ├── BERT-DINO-trainer.ipynb    # BERT + DINOv2 + numeric → price
│       ├── SBERT-DINO-trainer.ipynb   # Sentence-BERT + DINOv2 + numeric → price
│       ├── knn_sbert_trainer.py       # KNN-augmented SBERT model (FAISS neighbors)
│       └── ensemble.ipynb            # Average ensemble + SMAPE evaluation
└── data/                              # LFS-tracked CSVs (not included in repo)
```

## Pipeline

```
1. Split Data         →  create_splits.py
2. Feature Engineer   →  preprocess_stage1.py  →  preprocess_stage2.py
3. Download Images    →  download_images.py
4. Image Embeddings   →  create_dino_embeddings.py
5. Train Models       →  BERT-DINO / SBERT-DINO / KNN-SBERT trainers
6. Ensemble           →  ensemble.ipynb (simple average of predictions)
```

## Results

| Model | Val SMAPE |
|:---|:---|
| BERT + DINOv2 | 43.65% |
| SBERT + DINOv2 | 44.85% |
| KNN + SBERT | 46.48% |
| **Ensemble (avg)** | **41.5%** |

## Key Technical Details

- **Training**: PyTorch DDP (multi-GPU), mixed precision (AMP), cosine LR schedule with warmup
- **Text**: BERT pooler output (768-d) or SBERT sentence embedding (768-d), max_seq_length=128
- **Images**: DINOv2 ViT-B14 via `torch.hub`, 224×224 input, handles corrupted images with auto-redownload
- **KNN**: FAISS `IndexFlatIP` on SBERT embeddings, K=5 neighbors, neighbor prices + features concatenated
- **Preprocessing**: Two-stage unit normalization (kg→g→oz, L→mL→fl_oz), pack count extraction via regex, optional OpenAI fallback
- **Loss**: MSE on `log1p(price)`, predictions via `expm1`

## Reproduction

1. Place `train.csv` and `test.csv` in `data/`
2. Run preprocessing: `python code/preprocessing/create_splits.py` → `preprocess_stage1.py` → `preprocess_stage2.py`
3. Download images: `python code/images/download_images.py`
4. Generate embeddings: `python code/images/create_dino_embeddings.py --image-folder <path> --csv-path <path>`
5. Train models using the notebooks/scripts in `code/training/`
6. Run `ensemble.ipynb` on validation outputs
