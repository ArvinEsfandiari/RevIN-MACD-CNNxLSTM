# RevIN-MACD-CNNxLSTM

[![GitHub](https://img.shields.io/badge/GitHub-CNN--xLSTM-black?logo=github)](https://github.com/ArvinEsfandiari/RevIN-MACD-CNNxLSTM)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

**Wavelet, Genetic algorithm for optimizing MACD parameters and REverse Instance Normalization(RevIN) with CNNxLSTM as a predictor** — hybrid CNN-xLSTM forecasting with genetic algorithm–optimized MACD for cryptocurrency markets.

A research codebase for cryptocurrency time-series forecasting and algorithmic trading. It combines a hybrid **CNN-xLSTM** forecasting model with a **genetic algorithm–optimized MACD** indicator. Trades are taken only when both signals agree on direction (e.g., MACD buy + model predicts up → approved long).

Primary assets: **BTC** and **ETH** on **1h** and **4h** timeframes.

## Overview

The pipeline has four main stages:

1. **Data and Preprocessing** — Load OHLCV data (MetaTrader 5 or CSV), resample timeframes, and denoise with wavelet transforms (DWT). An EM-Kalman filter implementation is also available as an optional, experimental denoising method.
2. **MACD optimization** — Tune MACD parameters via genetic algorithms, brute-force search, or neuro-genetic hybrids.
3. **Forecasting** — Train CNN-xLSTM (with RevIN normalization) for directional price prediction. Fourier Analysis Normalization(FAN) is also implemented for expriment and comparing the results with RevIN.
4. **Strategy & backtesting** — Combine optimized MACD signals with model predictions and evaluate performance.

## Architecture

All process architecture at a glance is as a below:
<<<<<<< HEAD
[📄 All Process (PDF)](./src/fig/AllProcess_v2.jpg)

=======
<!-- [📄 All Process (PDF)](./src/fig/AllProcess_v2.pdf) -->
<object data="./fig/img/AllProcess_v2.pdf" type="application/pdf" width="100%" height="700px">
    <p>Your browser does not support embedding PDFs. <a href="./fig/img/AllProcess_v2.pdf">Download PDF instead</a>.</p>
</object>
>>>>>>> 3458fb30a968a03a916e2ed7d3b44a6dfaba5878

As you can see, it consists of many blocks:
1. Denoising
    1. Wavelet(db4, L1)
    2. Wavelet(db4, L2)
2. Genetic Algorithm
3. Prediction Process
4. Adjusting MACD and strategy

### Denoising
Although some denoising process had been tested, the best method among them was Wavelet L1 and L2 which L1 is used for Prediction Process and L2 is used for Genetic Algorithm. The process of denoising is as below:

[📄 Denoising Process and Blocks (PDF)](./fig/img/Wavelet_all_v1.pdf)

In addition, the decomposition block is shown below:
[📄 Decomposition of wavelet (PDF)](./fig/img/Wavelet_v3.pdf)







## Project Structure

```
src/
├── brute_force/     # Exhaustive MACD parameter search (incl. Kalman filter variants)
├── cnn/             # Indicator extraction and CNN-xLSTM evaluation notebooks
├── data/            # Data loading and exploration
├── denoise/         # Wavelet denoising (DWT) + optional EM-Kalman filter
├── FAN/             # FAN-xLSTM Colab training and local result analysis
├── genetic/         # Genetic algorithm MACD optimization
├── macdxlstm/       # End-to-end data prep: denoise → GA MACD → feature export
├── modelOnColab/    # Full wavelet → RevIN → CNN-xLSTM prediction workflow (Colab)
├── strategy/        # Combined MACD + xLSTM strategy backtesting
└── utils/           # Shared helpers (data loading, backtesting, metrics, preprocessing)
```

#### Notable notebooks

| Path | Role |
|------|------|
| `cnn/indicatorExtractor.ipynb` | Build the feature set (wavelet-denoised OHLC, RSI, MACD, Bollinger Bands, OBV, ADX, etc.) and load GA-optimized MACD features |
| `cnn/CNNxLSTM_error.ipynb` | Evaluate CNN-xLSTM directional prediction accuracy |
| `cnn/CNNxLSTM_fig.ipynb` | Generate model evaluation figures |
| `FAN/FANxLSTMModelColab.ipynb` | Train FAN-xLSTM on Google Colab; saves outputs to `fan_v1/` and `fan_v2/` |
| `FAN/fan_v1.ipynb` | Load saved FAN-xLSTM predictions from `fan_v1/` or `fan_v2/` and analyze results locally |
| `modelOnColab/fullPredictionModel.ipynb` | End-to-end Colab workflow: wavelet denoising → RevIN → CNN-xLSTM prediction |

### Key Python modules

| Module | Purpose |
|--------|---------|
| `utils/data_loader.py` | Fetch and save historical data from MetaTrader 5 |
| `utils/utils.py` | Timeframe resampling (`TFConvertor`, `CreateTimeFrames`) |
| `utils/backtester.py` | MACD strategy backtester with trading fees |
| `utils/directional_prediction.py` | Directional classification metrics and confusion matrices |
| `utils/coeffs2lines.py` | Convert GA MACD coefficients into indicator lines |
| `denoise/dwt.py` | Wavelet denoising — used in the main pipeline (reference: [10.1002/for.3071](https://doi.org/10.1002/for.3071)) |
| `denoise/EMkalman.py` | EM-Kalman filter denoising — optional/experimental; not wired into the main workflow |
| `genetic/algorithm.py` | `MACDOptimizer` — pygad-based MACD parameter search |
| `genetic/adaptiveGA_new.py` | Neuro-genetic hybrid (`MACDOptimizerGA_new`) |
| `genetic/classes.py` | GA backtesters, Kalman filters, and optimizer classes |
| `brute_force/classes.py` | MACD calculator and brute-force optimization utilities |
| `strategy/functions.py` | MACD crossover signal generation |

### Sample data

Preprocessed CSV files for strategy backtesting live in `src/strategy/files/`:

- `df_all_BTC_1h.csv`, `df_all_BTC_4h.csv`
- `df_all_ETH_1h.csv`, `df_all_ETH_4h.csv`

## Results

> Results below are reproduced from notebook outputs. Figures are generated at runtime by `strategy/strategy_resultsOK.ipynb` and saved to `src/strategy/figs/`.

### Strategy backtest — ETH 1h (`strategy/strategy_resultsOK.ipynb`)

| Strategy | Total Return | Annualized Return | Sharpe | Max Drawdown | Win Rate |
|----------|-------------:|------------------:|-------:|-------------:|---------:|
| Optimized MACD only | 29.74% | 12.86% | 7.64 | 7.83% | 63.64% |
| MACD + CNN-xLSTM | 29.16% | 12.70% | 6.89 | 5.91% | 82.14% |

The combined strategy reduces max drawdown and raises win rate by filtering MACD signals with model agreement, at a modest cost to Sharpe ratio.

**Generated figures** (after running the notebook):

- `src/strategy/figs/MACD_MARKET_ETH1h.png` — optimized MACD vs. buy-and-hold
- `src/strategy/figs/xLSTM_MACD_MARKET_ETH1h.png` — MACD-only vs. MACD + xLSTM

### Model accuracy — BTC 1h (`cnn/CNNxLSTM_error.ipynb`)

| Split | Accuracy |
|-------|---------:|
| Train | 66.36% |
| Validation | 68.91% |
| Test | 66.67% |

Additional test metrics: Recall 67.31%, F1 Score 67.48%.

## Workflow

Most work is done in Jupyter notebooks.

### Local pipeline

1. **`data/data_loader.ipynb`** or **`utils/data_loader.py`** — acquire OHLCV data.
2. **`macdxlstm/prepare_data.ipynb`** — wavelet denoise, run GA MACD optimization, export `.npy` features.
3. **`cnn/indicatorExtractor.ipynb`** — extract technical indicators and assemble the model feature matrix.
4. **`strategy/macd_strategy.ipynb`** / **`strategy/strategy_resultsOK.ipynb`** — backtest MACD-only vs. MACD + xLSTM combined strategies.

### Colab pipelines

Heavy model training is done on Google Colab and results are brought back locally:

- **`modelOnColab/fullPredictionModel.ipynb`** — full prediction stack: wavelet denoising → RevIN normalization → CNN-xLSTM.
- **`FAN/FANxLSTMModelColab.ipynb`** — trains the FAN-xLSTM model and writes predictions to `src/FAN/fan_v1/` and `src/FAN/fan_v2/`.
- **`FAN/fan_v1.ipynb`** — loads the saved `.pt` prediction files from those folders and runs directional evaluation locally.

Exploratory and experimental notebooks are also under `genetic/`, `brute_force/`, and `denoise/` (including `EMkalman.py`, which is available for integration but not used in the core pipeline).

## Trading Logic

Positions open or close only when **both** conditions align:

- **Optimized MACD** signals a buy or sell (crossover with tuned fast/slow/signal periods).
- **CNN-xLSTM** forecast agrees on direction (e.g., predicted price move matches signal).

Backtests account for exchange fees (defaults reference Nobitex maker/taker costs in `MACDBacktester`).

## Getting Started

### Prerequisites

- Python 3.12+
- (Optional) MetaTrader 5 terminal for live data fetching

### Installation

```bash
git clone git@github.com:ArvinEsfandiari/RevIN-MACD-CNNxLSTM.git
cd RevIN-MACD-CNNxLSTM
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

HTTPS alternative:

```bash
git clone https://github.com/ArvinEsfandiari/RevIN-MACD-CNNxLSTM.git
cd RevIN-MACD-CNNxLSTM
```

### Running notebooks

```bash
cd src
jupyter notebook
```

Run notebooks from the `src/` directory so relative imports (e.g. `from utils.utils import CreateTimeFrames`) resolve correctly.

> **Note:** `MetaTrader5` is only required if you fetch data via `utils/data_loader.py`. CSV-based workflows do not need it.

## References

| Topic | Reference |
|-------|-----------|
| xLSTM | Beck, M. et al. [*xLSTM: Extended Long Short-Term Memory*](https://arxiv.org/abs/2405.04517) (2024) |
| RevIN | Kim, T. et al. [*Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*](https://openreview.net/forum?id=cGDAkQo1C0p) (ICLR 2022) |
| FAN | [*Fourier Analysis Networks*](https://arxiv.org/abs/2410.02675) (2024) |
| Wavelet denoising | [doi:10.1002/for.3071](https://doi.org/10.1002/for.3071) |
| Genetic algorithm | [PyGAD](https://pygad.readthedocs.io/) — open-source GA library used for MACD parameter search |

## Author

**Arvin** — [arvin.esfandyari1377@gmail.com](mailto:arvin.esfandyari1377@gmail.com)

## Disclaimer

This repository is for **research and educational purposes only**. It is not financial advice. Cryptocurrency trading involves substantial risk of loss. Past backtest performance does not guarantee future results. Use at your own risk.

## License

MIT License — see [LICENSE](LICENSE). Copyright (c) 2024 Arvin.
