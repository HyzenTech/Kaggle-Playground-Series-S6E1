# 🏆 Kaggle Playground Series S6E1 - Student Test Scores

**Competition:** [Predicting Student Test Scores](https://www.kaggle.com/competitions/playground-series-s6e1)

## 📊 Results

| Version | CV RMSE | LB Score | Rank |
|---------|---------|----------|------|
| v5 | 8.71 | 8.66 | #260 |

## 📁 Repository Structure

```
├── Dataset/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── Code/
│   ├── professional_solution.ipynb    # Main Kaggle notebook
│   ├── exact_v5_kaggle.ipynb          # Minimal v5 notebook
│   ├── run_aggressive_v5.py           # Local Python script
│   └── submission_v5.csv              # Submission file
└── README.md
```

## 🔑 Key Techniques

1. **Original Dataset Augmentation** - Using the source dataset for cleaner patterns
2. **Magic Formula** - Linear combination: `5.91×study + 0.35×attendance + 1.42×sleep + 4.78`
3. **7-Model Ensemble** - Ridge, ElasticNet, BayesianRidge, ExtraTrees, LightGBM, XGBoost, CatBoost
4. **Hill Climbing Weights** - Optimization with negative weights allowed
5. **Ridge Stacking** - Meta-learner on OOF predictions

## 🚀 Quick Start

### Run on Kaggle
1. Upload `Code/professional_solution.ipynb`
2. Add the [original dataset](https://www.kaggle.com/datasets/kundanbedmutha/exam-score-prediction-dataset)
3. Enable GPU accelerator
4. Run all cells

### Run Locally
```bash
cd Code
python run_aggressive_v5.py
```

## 📈 Feature Engineering

- Trigonometric features (sin/cos)
- Polynomial features (log, square, cube, sqrt)
- Frequency encoding for categoricals
- Interaction features
- Ordinal encoding

## 🔧 Model Configuration

| Model | GPU | Iterations | Learning Rate |
|-------|-----|------------|---------------|
| LightGBM | CPU | 4000 | 0.012 |
| XGBoost | CUDA | 4000 | 0.012 |
| CatBoost | GPU | 4000 | 0.012 |

## 📜 License

MIT
