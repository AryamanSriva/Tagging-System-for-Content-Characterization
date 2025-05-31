"""
Text preprocessing utilities for the auto-tagging system.
"""

import re
import contractions
import spacy


class TextPreprocessor:
    """Handles all text preprocessing operations."""
    
    def __init__(self, spacy_model='en_core_web_md'):
        """Initialize the preprocessor with spaCy model."""
        self.nlp = spacy.load(spacy_model, disable=['parser', 'lemmatizer', 'attribute_ruler'])
    
    def convert_to_lowercase(self, text):
        """Convert text to lowercase."""
        return text.lower()
    
    def expand_contractions(self, text):
        """Expand contractions in text."""
        return contractions.fix(text)
    
    def remove_urls(self, text):
        """Remove URLs from text."""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, ' ', text)
    
    def remove_email_addresses(self, text):
        """Remove email addresses from text."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, ' ', text)
    
    def remove_dates_times(self, text):
        """Remove dates and times from text."""
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        time_pattern = r'\b\d{1,2}:\d{2}(?: \s?[AP]M)?\b'
        text = re.sub(date_pattern, ' ', text)
        text = re.sub(time_pattern, ' ', text)
        return text
    
    def remove_numbers_and_special_characters(self, text):
        """Remove numbers and special characters from text."""
        number_pattern = r'\b\d+\b'
        special_char_pattern = r'[^\w\s\.]|_|\s+'
        text = re.sub(number_pattern, ' ', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()
        return re.sub(special_char_pattern, ' ', text)
    
    def remove_stop_words_and_spaces(self, text):
        """Remove stop words and extra spaces."""
        doc = self.nlp(text)
        filtered_text = [token.text for token in doc if not token.is_stop]
        return ' '.join(filtered_text)
    
    def tokenize_text(self, text):
        """Tokenize text using spaCy."""
        tokens = [token.text for token in self.nlp(text)]
        return tokens
    
    def preprocess_data(self, text):
        """Apply all preprocessing functions in sequence."""
        text = self.convert_to_lowercase(text)
        text = self.expand_contractions(text)
        text = self.remove_urls(text)
        text = self.remove_email_addresses(text)
        text = self.remove_dates_times(text)
        text = self.remove_numbers_and_special_characters(text)
        text = self.remove_stop_words_and_spaces(text)
        # Clean up extra spaces
        text = re.sub(r'\s{2,}', ' ', text).strip()
        return text
    
    def preprocess_and_tokenize(self, text):
        """Preprocess text and return tokens."""
        processed_text = self.preprocess_data(text)
        return self.tokenize_text(processed_text)