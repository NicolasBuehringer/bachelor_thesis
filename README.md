# Realized Volatility Prediction using CNNs

This repository contains the code for my bachelor thesis. The project focuses on predicting realized volatility using high-frequency financial data and Convolutional Neural Networks (CNNs). It explores treating raw log-returns over a period as 2D spatial data representing the structure of financial market fluctuations.

## Project Structure

- `src/`: Core Python modules covering data preparation, modeling, and training.
  - `data_loader.py`: Handles raw CSV data ingestion.
  - `preprocessing.py`: Calculates log returns, realized volatility (RV), converts sequences into 21-day "image" blocks, and normalizes inputs.
  - `model.py`: Stores the CNN architecture constructed via Keras.
  - `trainer.py`: Oversees model compilation, ensemble training, and result persistence.
  - `utils.py`: Auxiliary tools primarily for generating evaluation metrics and training history plots.
- `notebooks/`: Contains the original Jupyter Notebook (`buehringer_RV_CNN.ipynb`) used for initial prototyping and visualization.
- `main.py`: The root level CLI to easily fire up full data processing and model training pipelines.

## Getting Started

### 1. Requirements

Ensure you have Python installed. Install the needed packages via:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Data

If using local data, place your raw high-frequency stock data (in `.csv` format) within a designated folder, such as `Data_10_year/` located at the directory root. Ensure your dataset aligns with the expected formatting (requires "Date Time" and "Close" columns).

### 3. Execution

Launch the whole sequence through the main script. You can pass the directory to the datasets. Wait for the predictions!
```bash
python main.py --data-path Data_10_year/
```
By default, trained neural networks persist into an `all_models/` folder, and the evaluation results and arrays flow into an `all_num_results/` folder. For customization, use:
```bash
python main.py --help
```

## Methodology

This approach fundamentally reimagines market volatility forecasting by leaning heavily into deep artificial neural networks:
1. **Sequence translation**: Rolling sets of 380 intraday log-returns across 21 days are gathered. 
2. **Channel division**: Negative stock movements map tightly to one channel while positive returns map to a separate channel.
3. **MinMax scaling**: The mapped data assumes ranges bound tightly between `[0, 255]` much like traditional pixels in visual tasks to support deep feature mapping.
4. **CNN Extraction**: Standard convolutions handle the high dimensional structured mappings, bypassing heavy traditional feature engineering.
