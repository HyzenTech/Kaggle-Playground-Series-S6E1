# 🏆 Kaggle Playground Series S6E1 - Predicting Student Test Scores

<div align="center">

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/playground-series-s6e1)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**🥈 Best Rank: #17 | 🎯 Target: Top 3**

[Competition Link](https://www.kaggle.com/competitions/playground-series-s6e1) • [Leaderboard](https://www.kaggle.com/competitions/playground-series-s6e1/leaderboard)

</div>

---

## 📊 Competition Overview

| Metric | Value |
|--------|-------|
| **Task** | Regression (Predict exam scores) |
| **Evaluation** | RMSE (Root Mean Squared Error) |
| **Train Size** | 195,469 samples |
| **Test Size** | 130,313 samples |
| **Features** | 14 (demographics, study habits, resources) |

---

## 🏅 Results & Progress

| Version | Description | CV RMSE | LB Score | Rank |
|---------|-------------|---------|----------|------|
| v1 | Baseline XGBoost | 8.85 | 8.72 | #400+ |
| v3 | Multi-model ensemble | 8.75 | 8.66 | #265 |
| v5 | Hill Climbing + Stacking | 8.71 | 8.55 | #50 |
| **v6** | **Advanced Blending (3 notebooks)** | - | **8.548** | **#17** |

---

## 📁 Repository Structure

```
📦 Kaggle-Playground-Series-S6E1
├── 📂 Code/
│   ├── professional_solution.ipynb    # Full pipeline with 7 models
│   ├── exact_v5_kaggle.ipynb          # Minimal v5 solution
│   ├── run_aggressive_v5.py           # Local execution script
│   └── submission_v5.csv              # Latest submission
├── 📂 Dataset/
│   ├── train.csv                      # Training data
│   ├── test.csv                       # Test data
│   └── sample_submission.csv          # Submission format
├── 🔬 blend_advanced_top3.ipynb       # Advanced blending notebook
├── 📋 kernel-metadata-advanced.json   # Kaggle metadata
└── 📖 README.md
```

---

## 🔬 Solution Architecture

### Phase 1: Feature Engineering
```
Raw Features → Transforms → 50+ Engineered Features
```

| Category | Features |
|----------|----------|
| **Polynomial** | log, square, cube, sqrt of numeric features |
| **Trigonometric** | sin/cos transformations |
| **Interaction** | study × attendance, sleep × study |
| **Encoding** | Ordinal encoding for categoricals |
| **Magic Formula** | `5.91×study + 0.35×attendance + 1.42×sleep + 4.78` |

### Phase 2: Multi-Model Training

| Model | Type | GPU/CPU | Key Params |
|-------|------|---------|------------|
| Ridge | Linear | CPU | α from CV |
| ElasticNet | Linear | CPU | l1_ratio=0.5 |
| BayesianRidge | Bayesian | CPU | Default |
| ExtraTrees | Ensemble | CPU | n_estimators=200 |
| LightGBM | GBDT | CPU | lr=0.012, iters=4000 |
| XGBoost | GBDT | CUDA | lr=0.012, iters=4000 |
| CatBoost | GBDT | GPU | lr=0.012, iters=4000 |

### Phase 3: Ensemble Strategy

```
┌─────────────────────────────────────────────────────────┐
│                    7 Base Models                        │
│  Ridge │ ElasticNet │ Bayesian │ Trees │ LGB │ XGB │ CB │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   OOF Predictions   │
              └──────────┬──────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌───────────┐   ┌───────────┐   ┌───────────┐
   │   Hill    │   │   Ridge   │   │  Simple   │
   │ Climbing  │   │  Stacking │   │  Average  │
   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
              ┌─────────────────────┐
              │   Final Blend      │
              │   (0.7 HC + 0.3 RC) │
              └─────────────────────┘
```

### Phase 4: Advanced Blending (Top 3 Strategy)

The `blend_advanced_top3.ipynb` implements:

| Technique | Formula | Use Case |
|-----------|---------|----------|
| **Arithmetic Mean** | `(p₁ + p₂ + ... + pₙ) / n` | Baseline |
| **Power Mean** | `((p₁ᵏ + p₂ᵏ + ...)/n)^(1/k)` | Emphasize extremes |
| **Geometric Mean** | `(p₁ × p₂ × ... × pₙ)^(1/n)` | Multiplicative blend |
| **Rank Average** | Convert to ranks → average | Different distributions |

---

## 🚀 Quick Start

### Option 1: Run on Kaggle (Recommended)

1. **Fork** the notebook: [S6E1 | Blend Top Public Notebooks](https://www.kaggle.com/code/muhammadhafizy/s6e1-blend-top-public-notebooks)
2. **Add inputs** (competition data is auto-added)
3. **Run All** and submit

### Option 2: Run Locally

```bash
# Clone repository
git clone https://github.com/HyzenTech/Kaggle-Playground-Series-S6E1.git
cd Kaggle-Playground-Series-S6E1

# Install dependencies
pip install numpy pandas scikit-learn lightgbm xgboost catboost

# Run solution
python Code/run_aggressive_v5.py

# Output: submission.csv
```

### Option 3: Advanced Blending

```bash
# On Kaggle, add these notebooks as inputs:
# - student-scores-from-lightgbm-to-senet
# - ps-s6e1-hb13g  
# - s6e1-hill-climbing-ridgecv-lb-8-54853

# Then run blend_advanced_top3.ipynb
```

---

## 📈 Key Insights

### What Worked ✅

1. **Original Dataset Augmentation** - Using the source Kaggle dataset alongside competition data
2. **Negative Weights in Hill Climbing** - Allowing weights from -0.5 to 1.5 for error cancellation
3. **Magic Formula** - The linear relationship `5.91×study + 0.35×attendance + 1.42×sleep + 4.78` captures 60%+ variance
4. **Multi-Notebook Blending** - Combining diverse public notebooks beats single-model training

### What Didn't Work ❌

1. Heavy feature selection - Keeping all features worked better
2. Deep neural networks - GBDTs dominated on this dataset
3. Simple averaging without optimization

---

## 🛠️ Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0
catboost>=1.0.0
scipy>=1.7.0
```

---

## 📚 References & Acknowledgements

This solution builds upon excellent public notebooks:

- [Student Scores | from LightGBM to SENet](https://www.kaggle.com/code/ambrosm/student-scores-from-lightgbm-to-senet) by AmbrosM
- [PS s6e1 | hb13g](https://www.kaggle.com/code/hb13g/ps-s6e1-hb13g) by hb13g
- [S6E1 - Hill Climbing & RidgeCV](https://www.kaggle.com/code/thomastschinkel/s6e1-hill-climbing-ridgecv-lb-8-54853) by Thomas Tschinkel

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for the Kaggle Community**

⭐ Star this repo if you found it helpful!

</div>
