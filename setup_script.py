#!/usr/bin/env python3
"""
Setup script to download required spaCy models and create necessary directories.
"""

import os
import subprocess
import sys


def run_command(command):
    """Run a command and return success status."""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def main():
    """Setup the auto-tagging system environment."""
    print("=== Auto-Tagging System Setup ===\n")
    
    # Create models directory
    print("1. Creating directories...")
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    print("✓ Directories created: models/, data/")
    
    # Install required packages
    print("\n2. Installing required packages...")
    success, output = run_command("pip install -r requirements.txt")
    if success:
        print("✓ Required packages installed successfully!")
    else:
        print(f"✗ Error installing packages: {output}")
        return False
    
    # Download spaCy model
    print("\n3. Downloading spaCy English model...")
    success, output = run_command("python -m spacy download en_core_web_md")
    if success:
        print("✓ spaCy English model downloaded successfully!")
    else:
        print(f"✗ Error downloading spaCy model: {output}")
        print("Please run manually: python -m spacy download en_core_web_md")
    
    # Download NLTK data (if needed)
    print("\n4. Setting up NLTK...")
    try:
        import nltk
        print("✓ NLTK is ready!")
    except ImportError:
        print("⚠ NLTK not found, but it's included in requirements.txt")
    
    print("\n=== Setup Complete! ===")
    print("\nNext steps:")
    print("1. Place your 'news_data.csv' file in the current directory")
    print("2. (Optional) Place 'test_data.csv' for evaluation")
    print("3. Run: python main.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)