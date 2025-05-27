"""
Entity extraction utilities using spaCy NER.
"""

import spacy
import random
from spacy.training import Example


class EntityExtractor:
    """Handles entity extraction and model training."""
    
    def __init__(self, spacy_model='en_core_web_md'):
        """Initialize the entity extractor with spaCy model."""
        self.nlp = spacy.load(spacy_model)
    
    def get_entities(self, text):
        """Extract entities from text using spaCy NER."""
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        return entities
    
    def extract_entities_pipe(self, texts):
        """Extract entities from multiple texts using spaCy pipe for efficiency."""
        entities = []
        for doc in self.nlp.pipe(texts):
            entities.append([(ent.text, ent.label_) for ent in doc.ents])
        return entities
    
    def aggregate_entities(self, entities_list):
        """Aggregate entities by label and remove duplicates."""
        aggregated_entities = {}
        for ent_text, ent_label in entities_list:
            if ent_label in aggregated_entities:
                aggregated_entities[ent_label].add(ent_text)
            else:
                aggregated_entities[ent_label] = {ent_text}
        return {label: list(texts) for label, texts in aggregated_entities.items()}
    
    def train_model(self, train_data, iterations=30):
        """
        Train the NER model with custom training data.
        
        Args:
            train_data (list): List of tuples (text, annotations)
            iterations (int): Number of training iterations
        """
        # Update the NER component with new examples
        ner = self.nlp.get_pipe('ner')
        for _, annotations in train_data:
            for ent in annotations.get('entities'):
                ner.add_label(ent[2])
        
        # Disable other pipeline components for training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'ner']
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.resume_training()
            for iteration in range(iterations):
                random.shuffle(train_data)
                for text, annotations in train_data:
                    doc = self.nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    self.nlp.update([example], drop=0.5, sgd=optimizer)
    
    def save_model(self, path):
        """Save the trained model to disk."""
        self.nlp.to_disk(path)
    
    def load_model(self, path):
        """Load a trained model from disk."""
        self.nlp = spacy.load(path)
    
    def get_default_training_data(self):
        """Return default training data for model improvement."""
        return [
            ("Noida is a city in India", {"entities": [(0, 5, "LOC")]}),
            ("Google launches a new AI research lab in Zurich", {"entities": [(0, 6, "ORG"), (37, 43, "LOC")]}),
            ("Amazon acquires Twitch for $970 million in 2014", {"entities": [(0, 6, "ORG"), (15, 21, "ORG"), (35, 42, "MONEY"), (46, 50, "DATE")]}),
            ("Elon Musk founded SpaceX to revolutionize space travel", {"entities": [(0, 9, "PERSON"), (18, 24, "ORG")]}),
            ("The Mona Lisa is on display in the Louvre Museum in Paris", {"entities": [(4, 13, "WORK_OF_ART"), (34, 47, "ORG"), (51, 56, "LOC")]}),
            ("The Great Wall of China stretches over 13,000 miles", {"entities": [(4, 22, "LOC"), (42, 51, "QUANTITY")]}),
            ("IBM introduces Watson, the AI that beat Jeopardy champions", {"entities": [(0, 3, "ORG"), (16, 22, "PERSON"), (31, 33, "ORG")]}),
            ("The Nile River flows through Egypt", {"entities": [(4, 14, "LOC"), (28, 33, "GPE")]}),
            ("Harvard University was established in 1636", {"entities": [(0, 17, "ORG"), (35, 39, "DATE")]}),
            ("Mount Everest is the world's highest mountain", {"entities": [(0, 13, "LOC")]}),
            ("Julia Roberts stars in the new Netflix series", {"entities": [(0, 12, "PERSON"), (31, 38, "ORG")]})
        ]