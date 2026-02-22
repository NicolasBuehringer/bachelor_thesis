import argparse
import sys
from src.trainer import run_and_save_model

def main():
    parser = argparse.ArgumentParser(description="Bachelor Thesis: Volatility Prediction using Convolutional Neural Networks")
    parser.add_argument("--data-path", type=str, default="Data_10_year/", help="Path to raw CSV data directory")
    parser.add_argument("--save-models", type=str, default="all_models", help="Directory to save trained models")
    parser.add_argument("--save-results", type=str, default="all_num_results", help="Directory to save numerical results")
    
    args = parser.parse_args()
    
    print(f"Starting pipeline with data from {args.data_path}")
    print(f"Saving models to: {args.save_models}")
    print(f"Saving results to: {args.save_results}")
    
    try:
        run_and_save_model(args.data_path, models_dir=args.save_models, results_dir=args.save_results)
        print("Pipeline finished successfully.")
    except Exception as e:
        print(f"Error occurred during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
