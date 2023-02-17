import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Open the CSV file and read the text data
with open('../S1_DatasetCollectors/telegram/datasetstg/tgyub.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip header row
    texts = [row[2] for row in reader] # assuming text data is in the second column

# Preprocess the text data
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove URLs, RTs, and twitter handles
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)

    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    return " ".join(tokens)

processed_texts = [preprocess_text(text) for text in texts]

# Save the processed text data to a new CSV file
with open('../S1_DatasetCollectors/telegram/datasetstg/preprocessed/tgyubprocessed_data.csv', 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['processed_text']) # write header row
    for text in processed_texts:
        writer.writerow([text])
