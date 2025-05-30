"""
Pattern matching utilities for entity recognition.
"""

import spacy
from spacy.matcher import Matcher


class PatternMatcher:
    """Handles pattern matching for specific entity types."""
    
    def __init__(self, spacy_model='en_core_web_md'):
        """Initialize the pattern matcher with spaCy model."""
        self.nlp = spacy.load(spacy_model)
        self.patterns = self._define_patterns()
    
    def _define_patterns(self):
        """Define patterns for entity matching."""
        return {
            "DATE_PATTERN": [
                {"IS_DIGIT": True, "LENGTH": 2, "OP": "?"},  # Day as two digits
                {"LOWER": {"IN": ["jan", "feb", "mar", "apr", "may", "jun", 
                                  "jul", "aug", "sep", "oct", "nov", "dec"]}},  # Month abbreviations
                {"IS_DIGIT": True, "LENGTH": 4, "OP": "?"}  # Four digit year
            ],
            "MONEY_PATTERN": [
                {"ORTH": "$"},  # Dollar sign
                {"LIKE_NUM": True},  # Numeric value
                {"LOWER": {"IN": ["thousand", "million", "billion"]}, "OP": "?"}  # Large amount modifiers
            ]
        }
    
    def find_matches(self, text, custom_patterns=None):
        """
        Find matches in text using defined patterns.
        
        Args:
            text (str): Text to search for patterns
            custom_patterns (dict): Custom patterns to use instead of default
            
        Returns:
            list: List of tuples containing (matched_text, pattern_label)
        """
        patterns_to_use = custom_patterns if custom_patterns else self.patterns
        
        # Create a Matcher object
        matcher = Matcher(self.nlp.vocab)
        
        # Add patterns to the matcher object
        for pattern_name, pattern in patterns_to_use.items():
            matcher.add(pattern_name, [pattern])
        
        doc = self.nlp(text)
        matches = matcher(doc)
        matched_entities = []
        
        for match_id, start, end in matches:
            matched_entities.append((doc[start:end].text, self.nlp.vocab.strings[match_id]))
        
        return matched_entities
    
    def add_pattern(self, pattern_name, pattern):
        """Add a new pattern to the existing patterns."""
        self.patterns[pattern_name] = pattern