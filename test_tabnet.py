"""
TabNet Installation and Quick Test
Verifies TabNet is properly installed and working.
"""

def check_tabnet_installation():
    """Check if TabNet and dependencies are installed."""
    print("="*80)
    print("TABNET INSTALLATION CHECK")
    print("="*80)
    
    checks = []
    
    # Check PyTorch
    try:
        import torch
        print(f"[OK] PyTorch installed: {torch.__version__}")
        checks.append(True)
    except ImportError:
        print("[FAIL] PyTorch not installed")
        print("       Install: pip install torch")
        checks.append(False)
    
    # Check TabNet
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        print("[OK] pytorch-tabnet installed")
        checks.append(True)
    except ImportError:
        print("[FAIL] pytorch-tabnet not installed")
        print("       Install: pip install pytorch-tabnet")
        checks.append(False)
    
    # Check NumPy
    try:
        import numpy as np
        print(f"[OK] NumPy installed: {np.__version__}")
        checks.append(True)
    except ImportError:
        print("[FAIL] NumPy not installed")
        checks.append(False)
    
    print("="*80)
    
    if all(checks):
        print("[SUCCESS] All TabNet dependencies installed!")
        return True
    else:
        print("[FAIL] Some dependencies missing")
        print("\nInstall all dependencies:")
        print("  pip install -r requirements.txt")
        return False

def test_tabnet_training():
    """Test TabNet training on small sample."""
    print("\n" + "="*80)
    print("TABNET TRAINING TEST")
    print("="*80)
    
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        import numpy as np
        import torch
        
        # Create small synthetic dataset
        np.random.seed(42)
        torch.manual_seed(42)
        
        X_train = np.random.randn(1000, 10)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
        
        print("Training TabNet on synthetic data (1000 samples, 10 features)...")
        
        model = TabNetClassifier(
            n_d=8, n_a=8, n_steps=3,
            gamma=1.3, lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            mask_type='entmax',
            verbose=0,
            seed=42
        )
        
        model.fit(
            X_train, y_train,
            max_epochs=10,
            batch_size=256,
            virtual_batch_size=128
        )
        
        # Test prediction
        X_test = np.random.randn(100, 10)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        print(f"[OK] Training complete")
        print(f"[OK] Predictions: {predictions[:5]}")
        print(f"[OK] Probabilities shape: {probabilities.shape}")
        print(f"[OK] Feature importances: {model.feature_importances_.shape}")
        
        print("\n[SUCCESS] TabNet is working correctly!")
        return True
        
    except Exception as e:
        print(f"[FAIL] TabNet test failed: {e}")
        return False

def main():
    """Run complete TabNet verification."""
    print("TabNet Installation and Testing")
    print("This will verify TabNet is ready for your FYP training pipeline\n")
    
    # Check installation
    if not check_tabnet_installation():
        print("\nPlease install missing dependencies before proceeding.")
        return False
    
    # Test training
    if not test_tabnet_training():
        print("\nTabNet training test failed. Check error messages above.")
        return False
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print("\n[SUCCESS] TabNet is ready!")
    print("\nYou can now run:")
    print("  python train_models.py")
    print("\nTabNet will be automatically included as the 5th model.")
    
    return True

if __name__ == "__main__":
    main()
