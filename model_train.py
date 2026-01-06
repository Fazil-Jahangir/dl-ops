# re - Regular Expression
# used to clean the text
# "Hello!!!!! :) " ---------- "hello"
import re


# used to "save model" to "file"
# After training, we don't want to repeat again and again
import joblib


#import nltk
#NLP library - deep learning library
import nltk
#punkit - sentence splitter
nltk.download('punkt')
#Identify stop words
nltk.download('stopwords')
#English Dictionary
nltk.download('wordnet')
nltk.download('punkt_tab')

#I love NLP
#i love nlp
#["i", "love", "nlp"]
# pip install nltk
# convert statements to words
from nltk.tokenize import word_tokenize


#identifies the stopwords
from nltk.corpus import stopwords


#converts words to dictionary root
#coverts normal english to dictionary
#playing ---> play
#better ---> good
from nltk.stem import WordNetLemmatizer


#convert your text ---> numbers
#meaningful words ---> high priority
from sklearn.feature_extraction.text import TfidfVectorizer

 
#LogisticRegression is the ML Algorithm
#Output : 0 (or). 1
from sklearn.linear_model import LogisticRegression


# Training Data
texts = ["I love this product", 
         "This product is very bad",
         "Amazing quality and good service",
         "Worst experience ever"]
labels = [1, 0, 1, 0]

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#Step 1 : Cleaning Process
cleaned_texts = []

for sentence in texts:
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-z\s]',  '', sentence)
    tokens = word_tokenize(sentence)
    words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    cleaned_texts.append(sentence)
#print(cleaned_texts)

#print(final_texts)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_texts)

model = LogisticRegression()
model.fit(X, labels)

joblib.dump(model, 'my_model.pkl')
joblib.dump(vectorizer, "vectorizer.pkl")