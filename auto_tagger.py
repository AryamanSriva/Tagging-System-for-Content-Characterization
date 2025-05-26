"""
Main auto-tagging system that combines all components.
"""

import pandas as pd
import ast
from text_preprocessor import TextPreprocessor
from pattern_matcher import PatternMatcher
from entity_extractor import EntityExtractor
from evaluator import Evaluator


class AutoTagger:
    """Main auto-tagging system for content categorization."""
    
    def __init__(self, spacy_model='en_core_web_md'):
        """Initialize the auto-tagging system."""
        self.preprocessor = TextPreprocessor(spacy_model)
        self.pattern_matcher = PatternMatcher(spacy_model)
        self.entity_extractor = EntityExtractor(spacy_model)
        self.evaluator = Evaluator()
    
    def process_single_text(self, text):
        """
        Process a single text and extract entities.
        
        Args:
            text (str): Input text to process
            
        Returns:
            dict: Dictionary containing processed results
        """
        # Preprocess text
        processed_text = self.preprocessor.preprocess_data(text)
        
        # Extract entities
        entities = self.entity_extractor.extract_entities_pipe([processed_text])[0]
        refined_entities = self.entity_extractor.aggregate_entities(entities)
        
        # Find pattern matches
        pattern_matches = self.pattern_matcher.find_matches(text)
        
        return {
            'original_text': text,
            'processed_text': processed_text,
            'entities': entities,
            'refined_entities': refined_entities,
            'pattern_matches': pattern_matches
        }
    
    def process_dataframe(self, df, text_column='News Data'):
        """
        Process a DataFrame containing text data.
        
        Args:
            df (pandas.DataFrame): DataFrame with text data
            text_column (str): Name of the column containing text
            
        Returns:
            pandas.DataFrame: DataFrame with added processing columns
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Preprocess all texts
        result_df['Processed_Data'] = result_df[text_column].apply(
            self.preprocessor.preprocess_data
        )
        
        # Tokenize texts
        result_df['Processed_Tokenized_Data'] = result_df['Processed_Data'].apply(
            self.preprocessor.tokenize_text
        )
        
        # Extract entities
        result_df['entities'] = self.entity_extractor.extract_entities_pipe(
            result_df['Processed_Data']
        )
        
        # Aggregate entities
        result_df['Refined_Entities'] = result_df['entities'].apply(
            self.entity_extractor.aggregate_entities
        )
        
        return result_df
    
    def train_custom_model(self, training_data=None, iterations=30, save_path=None):
        """
        Train a custom NER model.
        
        Args:
            training_data (list): Custom training data or None for default
            iterations (int): Number of training iterations
            save_path (str): Path to save the trained model
        """
        if training_data is None:
            training_data = self.entity_extractor.get_default_training_data()
        
        self.entity_extractor.train_model(training_data, iterations)
        
        if save_path:
            self.entity_extractor.save_model(save_path)
    
    def evaluate_performance(self, test_df, actual_col='Actual_Entities', 
                           predicted_col='Refined_Entities'):
        """
        Evaluate the performance of the auto-tagging system.
        
        Args:
            test_df (pandas.DataFrame): Test dataset
            actual_col (str): Column with actual entities
            predicted_col (str): Column with predicted entities
            
        Returns:
            dict: Evaluation metrics
        """
        return self.evaluator.evaluate_dataset(test_df, actual_col, predicted_col)
    
    def prepare_test_data(self, test_csv_path, text_column='News Data', 
                         actual_entities_column='Actual_Entities'):
        """
        Prepare test data from CSV file.
        
        Args:
            test_csv_path (str): Path to test CSV file
            text_column (str): Name of text column
            actual_entities_column (str): Name of actual entities column
            
        Returns:
            pandas.DataFrame: Prepared test DataFrame
        """
        df_test = pd.read_csv(test_csv_path)
        
        # Fix the Actual_Entities column if it's string representation of dict
        def dict_fix(val):
            if isinstance(val, str):
                return ast.literal_eval(val)
            return val
        
        df_test[actual_entities_column] = df_test[actual_entities_column].apply(dict_fix)
        
        # Process the test data
        df_test = self.process_dataframe(df_test, text_column)
        
        # Clean up intermediate columns
        df_test.drop(["Processed_Data", "entities"], axis=1, inplace=True, errors='ignore')
        
        return df_test


def main():
    """Example usage of the AutoTagger system."""
    # Initialize the auto-tagger
    tagger = AutoTagger()
    
    # Example: Process a single text
    sample_text = "Google launches a new AI research lab in Zurich on January 15, 2024."
    result = tagger.process_single_text(sample_text)
    
    print("Single Text Processing Result:")
    print(f"Original: {result['original_text']}")
    print(f"Processed: {result['processed_text']}")
    print(f"Entities: {result['refined_entities']}")
    print(f"Pattern Matches: {result['pattern_matches']}")
    print()
    
    # Example: Train custom model
    print("Training custom model...")
    tagger.train_custom_model(save_path='./models/custom_spacy_model')
    print("Model training completed!")
    

if __name__ == "__main__":
    main()