#!/usr/bin/env python3
"""
Main script to run the complete auto-tagging workflow.
This script replicates the functionality from the original Jupyter notebook.
"""

import pandas as pd
from auto_tagger import AutoTagger
from evaluator import Evaluator


def main():
    """Run the complete auto-tagging workflow."""
    print("=== Auto-Tagging System for Content Categorization ===\n")
    
    # Initialize the auto-tagger
    print("1. Initializing Auto-Tagger System...")
    tagger = AutoTagger()
    print("✓ System initialized successfully!\n")
    
    # Load and explore the dataset
    print("2. Loading and exploring the dataset...")
    try:
        df = pd.read_csv('news_data.csv')
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Shape: {df.shape}")
        print(f"  - Columns: {list(df.columns)}")
        
        # Display first row as example
        if 'News Data' in df.columns:
            print(f"  - Sample text: {df['News Data'].iloc[0][:100]}...")
        print()
    except FileNotFoundError:
        print("✗ Error: news_data.csv not found. Please ensure the file exists.")
        return
    
    # Process the main dataset
    print("3. Processing text data...")
    try:
        processed_df = tagger.process_dataframe(df)
        print("✓ Text processing completed!")
        print(f"  - Added columns: {[col for col in processed_df.columns if col not in df.columns]}")
        
        # Show sample processed result
        if len(processed_df) > 0:
            sample_entities = processed_df['Refined_Entities'].iloc[0]
            print(f"  - Sample entities extracted: {sample_entities}")
        print()
    except Exception as e:
        print(f"✗ Error during text processing: {str(e)}")
        return
    
    # Train custom model
    print("4. Training custom NER model...")
    try:
        tagger.train_custom_model(iterations=30, save_path='./models/spacy_optimised')
        print("✓ Custom model training completed!")
        print("  - Model saved to: ./models/spacy_optimised")
        
        # Test the trained model
        test_text = "Noida is a city"
        result = tagger.process_single_text(test_text)
        print(f"  - Test result for '{test_text}': {result['refined_entities']}")
        print()
    except Exception as e:
        print(f"✗ Error during model training: {str(e)}")
        print()
    
    # Evaluate on test data
    print("5. Evaluating model performance...")
    try:
        test_df = tagger.prepare_test_data('test_data.csv')
        print("✓ Test data prepared successfully!")
        
        # Calculate evaluation metrics
        metrics = tagger.evaluate_performance(test_df)
        print("✓ Evaluation completed!")
        print("\n=== Performance Metrics ===")
        Evaluator.print_evaluation_results(metrics)
        print()
        
    except FileNotFoundError:
        print("⚠ Warning: test_data.csv not found. Skipping evaluation.")
        print("  You can still use the system for processing new texts.")
        print()
    except Exception as e:
        print(f"✗ Error during evaluation: {str(e)}")
        print()
    
    # Demonstrate pattern matching
    print("6. Demonstrating pattern matching...")
    sample_text = "I invested $1000 on 11 Jan 2021 and it grew to $5 million by Dec 2022."
    result = tagger.process_single_text(sample_text)
    print(f"Sample text: {sample_text}")
    print(f"Pattern matches found: {result['pattern_matches']}")
    print(f"All entities: {result['refined_entities']}")
    print()
    
    print("=== Auto-Tagging System Demo Completed Successfully! ===")
    print("\nTo use this system in your own code:")
    print("1. from auto_tagger import AutoTagger")
    print("2. tagger = AutoTagger()")
    print("3. result = tagger.process_single_text('your text here')")


if __name__ == "__main__":
    main()