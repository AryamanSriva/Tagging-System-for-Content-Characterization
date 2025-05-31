# Auto-Tagging System for Content Categorization

A comprehensive Natural Language Processing (NLP) system for automatic content categorization and entity extraction from news articles and text documents.

## 🚀 Features

- **Text Preprocessing**: Comprehensive text cleaning including URL removal, contraction expansion, and special character handling
- **Entity Extraction**: Advanced Named Entity Recognition (NER) using spaCy models
- **Pattern Matching**: Custom pattern matching for dates, monetary values, and other specific entities
- **Model Training**: Custom NER model training with your own labeled data
- **Performance Evaluation**: Comprehensive metrics including precision, recall, and F1-score
- **Batch Processing**: Efficient processing of large datasets using spaCy's pipe functionality

## 📁 Project Structure

```
auto-tagging-system/
├── auto_tagger.py           # Main system orchestrator
├── text_preprocessor.py     # Text preprocessing utilities
├── pattern_matcher.py       # Pattern matching for specific entities
├── entity_extractor.py      # NER and entity extraction
├── evaluator.py            # Performance evaluation metrics
├── main.py                 # Main script to run complete workflow
├── setup.py               # Setup and installation script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── models/               # Directory for trained models
│   └── spacy_optimised/  # Custom trained spaCy model
└── data/                # Directory for data files
    ├── news_data.csv    # Main training dataset
    └── test_data.csv    # Test dataset for evaluation
```

## 🚀 Quick Start

### Basic Usage

```python
from auto_tagger import AutoTagger

# Initialize the system
tagger = AutoTagger()

# Process a single text
text = "Google launches a new AI research lab in Zurich on January 15, 2024."
result = tagger.process_single_text(text)

print("Entities found:", result['refined_entities'])
print("Pattern matches:", result['pattern_matches'])
```

### Processing a Dataset

```python
import pandas as pd
from auto_tagger import AutoTagger

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize and process
tagger = AutoTagger()
processed_df = tagger.process_dataframe(df, text_column='your_text_column')

# View results
print(processed_df[['original_text', 'Refined_Entities']].head())
```

### Training a Custom Model

```python
# Define your training data
training_data = [
    ("Apple Inc. is based in Cupertino", {"entities": [(0, 10, "ORG"), (23, 32, "GPE")]}),
    ("Tim Cook is the CEO of Apple", {"entities": [(0, 8, "PERSON"), (23, 28, "ORG")]})
]

# Train the model
tagger.train_custom_model(training_data, iterations=50, save_path='./models/my_model')
```

## 📊 Data Format

### Input Data Format

Your CSV file should contain a text column. Example:

```csv
News Data
"Apple Inc. reported strong quarterly earnings, with CEO Tim Cook highlighting growth..."
"Tesla's new factory in Berlin will produce electric vehicles for the European market..."
```

### Test Data Format (for evaluation)

```csv
News Data,Actual_Entities
"Apple Inc. in Cupertino...","{""ORG"": [""Apple Inc.""], ""GPE"": [""Cupertino""]}"
```

## 🔧 Configuration

### Customizing Text Preprocessing

```python
from text_preprocessor import TextPreprocessor

preprocessor = TextPreprocessor()

# Individual preprocessing steps
text = preprocessor.convert_to_lowercase(text)
text = preprocessor.expand_contractions(text)
text = preprocessor.remove_urls(text)
# ... etc

# Or use the complete pipeline
processed_text = preprocessor.preprocess_data(text)
```

### Adding Custom Patterns

```python
from pattern_matcher import PatternMatcher

matcher = PatternMatcher()

# Add a custom pattern for phone numbers
phone_pattern = [
    {"SHAPE": "ddd"},
    {"ORTH": "-"},
    {"SHAPE": "ddd"},
    {"ORTH": "-"},
    {"SHAPE": "dddd"}
]

matcher.add_pattern("PHONE_PATTERN", phone_pattern)
```

## 📈 Performance Evaluation

The system provides comprehensive evaluation metrics:

```python
from evaluator import Evaluator

# Evaluate on test data
metrics = tagger.evaluate_performance(test_df)

# Print results
Evaluator.print_evaluation_results(metrics)
```

**Output:**
```
Average Precision: 0.8542
Average Recall: 0.7891
Average F1 Score: 0.8201
```

## 🎯 Use Cases

- **News Categorization**: Automatically tag news articles with relevant entities
- **Content Management**: Organize large document collections
- **Social Media Analysis**: Extract entities from social media posts
- **Research Paper Analysis**: Identify key entities in academic papers
- **Legal Document Processing**: Extract important entities from legal texts

## 🧪 Example Results

### Input Text:
```
"Google announced a $2.1 billion acquisition of Fitbit on November 1, 2019, 
pending regulatory approval from the European Commission."
```

### Extracted Entities:
```python
{
    'ORG': ['Google', 'Fitbit', 'European Commission'],
    'MONEY': ['$2.1 billion'],
    'DATE': ['November 1, 2019']
}
```

### Pattern Matches:
```python
[
    ('$2.1 billion', 'MONEY_PATTERN'),
    ('November 1, 2019', 'DATE_PATTERN')
]
```

## 🔬 Advanced Features

### Batch Processing for Large Datasets

```python
# Process large datasets efficiently
large_df = pd.read_csv('large_dataset.csv')
processed_df = tagger.process_dataframe(large_df)

# The system uses spaCy's pipe for efficient batch processing
```

### Custom Entity Labels

```python
# Add custom entity types during training
custom_training_data = [
    ("Bitcoin price reached $50,000", {"entities": [(0, 7, "CRYPTOCURRENCY"), (21, 28, "MONEY")]})
]

tagger.train_custom_model(custom_training_data)
```

## 🐛 Troubleshooting

### Common Issues

1. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_md
   ```

2. **Memory issues with large datasets**
   - Process data in chunks
   - Use the batch processing functionality
   - Consider using the smaller `en_core_web_sm` model

3. **Poor entity extraction performance**
   - Train with more domain-specific data
   - Adjust preprocessing parameters
   - Use larger spaCy models

### Performance Tips

- Use `nlp.pipe()` for batch processing instead of individual texts
- Disable unused spaCy pipeline components
- Consider using GPU acceleration for large-scale processing

## 📋 Requirements

- **Python**: 3.7+
- **spaCy**: 3.4.0+
- **pandas**: 1.3.0+
- **contractions**: 0.1.0+
- **nltk**: 3.7+


## 🚀 Future Enhancements

- Support for multilingual text processing
- Integration with transformer models (BERT, RoBERTa)
- Web interface for easy interaction
- API endpoint for remote processing
- Docker containerization
- Real-time processing capabilities

---
