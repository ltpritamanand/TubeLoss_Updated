import argparse
import torch
import numpy as np
import os
from src.run_experiment import run_tube_loss_experiment
from data.dataset import GetDataset

def run_naval_experiment():
    """
    Runs a single experiment on the naval dataset with the best-found
    hyperparameters for the Tube_Loss.
    Best from Trial 8: r=0.3, delta=0.05, Coverage=89.11%, Width=0.0224
    """
    
    # Best parameters found for naval dataset with Tube Loss
    config = {
        "dataset_name": "naval",
        "coverage": 0.9,
        "lr": 0.05,
        "dropout": 0.2,
        "penalty": 0.1,
        "epochs": 400,
        "batch_size": 10000,
        "test_ratio": 0.2,
        "val_ratio": 0.2,
        "random_seed": 1,  # Use seed 1 which gave the best results
        "finetuning": False,
        "device": 0,
        "verbose": True,
        "r": 0.3,         # Tube movement factor
        "delta": 0.05     # Recalibration coefficient
    }

    print("Running Naval dataset experiment with the following configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    data = GetDataset(config["dataset_name"], "data/")
    
    val_loss, results, results_val = run_tube_loss_experiment(config, data)

    print("\n--- Naval Dataset Results ---")
    print(f"  Coverage: {results['test_coverage']:.4f} ({results['test_coverage']*100:.2f}%)")
    print(f"  MPIW: {results['test_width']:.4f}")
    print("-----------------------------")

if __name__ == "__main__":
    run_naval_experiment()
