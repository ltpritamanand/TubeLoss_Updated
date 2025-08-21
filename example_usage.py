#!/usr/bin/env python3
"""
Simple example demonstrating Tube Loss usage for conformal prediction.

This script shows how to:
1. Create synthetic data
2. Set up a Tube Loss model
3. Train the model
4. Evaluate prediction intervals
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightning as L

from src.model import TubeLossModel
from data.dataset import Dataset


def generate_synthetic_data(n_samples=1000, noise_level=0.1):
    """
    Generate synthetic regression data for demonstration.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    noise_level : float
        Standard deviation of noise to add
        
    Returns
    -------
    X : ndarray
        Input features
    y : ndarray
        Target values
    """
    np.random.seed(42)
    
    # Generate features
    X = np.random.uniform(-3, 3, (n_samples, 1))
    
    # Generate targets with non-linear relationship
    y_clean = 2 * np.sin(X[:, 0]) + 0.5 * X[:, 0]**2
    y = y_clean + noise_level * np.random.normal(0, 1, n_samples)
    
    return X.astype(np.float32), y.reshape(-1, 1).astype(np.float32)


def main():
    """Run the Tube Loss example."""
    
    print("🚀 Tube Loss Example")
    print("=" * 50)
    
    # 1. Generate synthetic data
    print("📊 Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=800, noise_level=0.2)
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # 3. Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 4. Normalize targets
    mean_y = np.mean(np.abs(y_train))
    y_train = y_train / mean_y
    y_val = y_val / mean_y
    y_test = y_test / mean_y
    
    print(f"   Train shape: {X_train.shape}")
    print(f"   Validation shape: {X_val.shape}")
    print(f"   Test shape: {X_test.shape}")
    
    # 5. Create datasets and dataloaders
    train_dataset = Dataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = Dataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_dataset = Dataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )
    
    # 6. Create Tube Loss model
    print("\n🧠 Creating Tube Loss model...")
    model = TubeLossModel(
        coverage=0.9,          # 90% coverage target
        x_shape=X_train.shape[1],  # Number of input features
        hidden_size=64,        # Hidden layer size
        dropout=0.1,           # Dropout rate
        lr=0.001,             # Learning rate
        r=0.5,                # Tube movement factor
        delta=0.05            # Width regularization
    )
    
    print(f"   Target coverage: {model.coverage:.1%}")
    print(f"   Tube movement factor (r): {model.loss_fn.r_value}")
    print(f"   Width regularization (δ): {model.loss_fn.delta_value}")
    
    # 7. Train the model
    print("\n🏋️ Training model...")
    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    # 8. Evaluate on test set
    print("\n📈 Evaluating on test set...")
    test_results = trainer.test(model, test_loader, verbose=False)
    
    coverage = test_results[0]['test_coverage']
    width = test_results[0]['test_width']
    
    print(f"   ✅ Test Coverage: {coverage:.3f} ({coverage*100:.1f}%)")
    print(f"   📏 Mean Interval Width: {width:.3f}")
    print(f"   🎯 Target Coverage: {model.coverage:.3f} ({model.coverage*100:.1f}%)")
    
    # 9. Generate predictions for visualization
    print("\n📊 Generating predictions...")
    model.eval()
    with torch.no_grad():
        # Get predictions on test set
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        predictions = model(X_test_tensor).numpy()
        
        lower_bounds = predictions[:, 0] * mean_y
        upper_bounds = predictions[:, 1] * mean_y
        
    # Sort for plotting
    sort_idx = np.argsort(X_test[:, 0])
    X_test_sorted = X_test[sort_idx, 0] 
    y_test_sorted = (y_test[sort_idx, 0] * mean_y)
    lower_sorted = lower_bounds[sort_idx]
    upper_sorted = upper_bounds[sort_idx]
    
    # Calculate actual coverage
    in_interval = (y_test_sorted >= lower_sorted) & (y_test_sorted <= upper_sorted)
    actual_coverage = np.mean(in_interval)
    
    print(f"   📊 Actual coverage on test set: {actual_coverage:.3f} ({actual_coverage*100:.1f}%)")
    
    # 10. Create visualization
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot prediction intervals
        plt.fill_between(
            X_test_sorted, lower_sorted, upper_sorted, 
            alpha=0.3, color='lightblue', label='90% Prediction Interval'
        )
        
        # Plot actual values
        plt.scatter(
            X_test_sorted, y_test_sorted, 
            alpha=0.6, s=20, color='red', label='True Values'
        )
        
        # Plot interval bounds
        plt.plot(X_test_sorted, lower_sorted, '--', color='blue', alpha=0.7, label='Lower Bound')
        plt.plot(X_test_sorted, upper_sorted, '--', color='blue', alpha=0.7, label='Upper Bound')
        
        plt.xlabel('Input Feature')
        plt.ylabel('Target Value')
        plt.title(f'Tube Loss Prediction Intervals\nCoverage: {actual_coverage:.1%}, Width: {np.mean(upper_sorted - lower_sorted):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig('tube_loss_example.png', dpi=150, bbox_inches='tight')
        print(f"   💾 Visualization saved as 'tube_loss_example.png'")
        plt.show()
        
    except ImportError:
        print("   ⚠️  Matplotlib not available for visualization")
    
    print("\n✅ Example completed successfully!")
    print("\nKey takeaways:")
    print(f"   • Tube Loss achieved {actual_coverage:.1%} coverage (target: {model.coverage:.1%})")
    print(f"   • Mean interval width: {np.mean(upper_sorted - lower_sorted):.3f}")
    print(f"   • The model learned to balance coverage and efficiency")


if __name__ == "__main__":
    main()
