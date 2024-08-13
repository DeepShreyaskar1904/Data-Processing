import pandas as pd
import nltk
import re
import string

nltk.download('stopwords')
from nltk.corpus import stopwords

data = {
    'text': [
        'I love this product! It works great.',
        'This is the worst experience I have ever had...',
        'Amazing quality, would definitely recommend!!!',
        'Terrible product, completely broke after one use.',
        'I am so happy with my purchase; it’s fantastic!'
    ],
    'label': ['positive', 'negative', 'positive', 'negative', 'positive']
}

df = pd.DataFrame(data)

print("Raw Data:")
print(df)

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
   
    text = text.translate(str.maketrans('', '', string.punctuation))
   
    text = text.lower()
   
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

print("\nCleaned Data:")
print(df[['text', 'cleaned_text', 'label']])

df['label'] = df['label'].map({'positive': 1, 'negative': 0})


X = df['cleaned_text']  
y = df['label']         


print("\nFinal Prepared Data:")
print("Features (X):")
print(X)
print("Target (y):")
print(y)
