"""
Dependency Check Script
Verifies all required packages are installed before running the model training pipeline.
"""

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'joblib': 'joblib',
        'imblearn': 'imbalanced-learn',
        'shap': 'shap'
    }
    
    missing = []
    installed = []
    
    print("Checking dependencies...")
    print("="*60)
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package}")
            installed.append(package)
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    print("="*60)
    
    if missing:
        print(f"\n⚠ Missing {len(missing)} package(s):")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall missing packages with:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print(f"\n✓ All {len(installed)} required packages are installed!")
        print("Ready to run: python train_models.py")
        return True

if __name__ == "__main__":
    check_dependencies()
