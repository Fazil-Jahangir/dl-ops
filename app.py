#import fastapi
#alternate to flask. make Post, Get, --> fast
from fastapi import FastAPI
# optional library ---> on top we will build api's
from pydantic import BaseModel
# load pkl files
import joblib
# import regular expression > clean data
import re
# import nlp library
import nltk
# remove stopwords like is, an, i, am
from nltk.corpus import stopwords
# statement --> words
from nltk.tokenize import word_tokenize
# words ---> dictionary level words
from nltk.stem import WordNetLemmatizer



#punkit - sentence splitter
nltk.download('punkt')
#Identify stop words
nltk.download('stopwords')
#English Dictionary
nltk.download('wordnet')
nltk.download('punkt_tab')


app = FastAPI()


model = joblib.load("my_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text:str):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(words)


class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message":"NLP FastAPI is running"}


@app.post("/predict")
def predict_sentiment(data: TextInput):
    cleaned_text = preprocess_text(data.text)
    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)

    result = "Positive" if prediction[0] == 1 else "Negative"

    return {
        "input_text": data.text,
        "sentiment": result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)