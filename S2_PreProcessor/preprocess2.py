import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from spellchecker import SpellChecker
import emoji
#from ekphrasis.classes.tokenizer import SocialTokenizer

nltk.download('vader_lexicon')
nltk.download('punkt')

text = 'processed_text'

#tokenizer = SocialTokenizer(lowercase=False)
tokens = nltk.word_tokenize(text)

# Load your preprocessed data
data = pd.read_csv('../S1_DatasetCollectors/telegram/datasetstg/tgpohenprocessed_data.csv')

# Load abbreviations_list
with open('abbreviations_list.txt', 'r', encoding='utf-8') as f:
    abbreviations_list = [line.strip() for line in f]

# Create a SentimentIntensityAnalyzer instance
sid = SentimentIntensityAnalyzer()

# Create a SpellChecker instance
spell = SpellChecker()

# Add new columns for the labels
data['sentence_length'] = None
data['vocabulary_complexity'] = None
data['use_of_punctuation'] = None
data['tone'] = None
data['specific_words'] = None
data['grammar'] = None
data['use_of_contractions'] = None
data['frequency_of_capitalization'] = None
data['use_of_emojis'] = None
data['use_of_abbreviations'] = None

# Loop through the data and add the labels
for i, row in data.iterrows():
    text = row['processed_text']
    
    # Sentence length
    sentence_length = len(nltk.sent_tokenize(str(text))) if isinstance(text, str) else 0
    data.at[i, 'sentence_length'] = sentence_length
    
    # Vocabulary complexity
    vocab_complexity = len(set(tokens)) / len(tokens)
    data.at[i, 'vocabulary_complexity'] = vocab_complexity
    
    # Use of punctuation
    text = str(text)
    punctuation_count = sum([1 for char in text if char in '.,?!'])
    data.at[i, 'use_of_punctuation'] = punctuation_count
    
    # Tone
    polarity_scores = sid.polarity_scores(text)
    tone = max(polarity_scores, key=polarity_scores.get)
    data.at[i, 'tone'] = tone
    
    # Specific words
    specific_words_count = len([w for w in tokens if w in ['word1', 'word2', 'word3']])
    data.at[i, 'specific_words'] = specific_words_count
    
    # Grammar
    grammar_errors_count = len(spell.unknown(tokens))
    data.at[i, 'grammar'] = grammar_errors_count
    
    # Use of contractions
    contractions_count = len([w for w in tokens if "'" in w])
    data.at[i, 'use_of_contractions'] = contractions_count
    
    # Frequency of capitalization
    capitalization_count = sum([1 for char in text if char.isupper()])
    data.at[i, 'frequency_of_capitalization'] = capitalization_count
    
    # Use of emojis
    emojis_count = len([char for char in text if char in emoji.EMOJI_DATA])
    data.at[i, 'use_of_emojis'] = emojis_count
    
    # Use of abbreviations
    abbreviations_count = len([w for w in tokens if w.upper() in abbreviations_list])
    data.at[i, 'use_of_abbreviations'] = abbreviations_count
data.to_csv('tgpohenprocessed_updated_data.csv', index=False)