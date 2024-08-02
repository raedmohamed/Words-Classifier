from nltk.corpus import wordnet

def nltk_tag_to_wordnet_tag(nltk_tag):
    tag_mapping = {
        "VERB": {"MD","VB","VBD","VBG","VBN","VBP","VBZ","HV","HVD","HVG","HVN","HVP","HVZ","BE","BED","BEG","BEN","BEP",
                "BEZ", },
        "ADJ": {"JJ","JJR","JJS","PDT","AFX","DT","WDT","PRP","PRP$","WP","WP$","WRB",},
        "NOUN": {"CD","NN","NNS","NNP","NNPS","SYM","TO","UH","EX","FW","IN","POS","RP","WDT","WP","WP$","WRB",},
        "ADV": {"RB", "RBR", "RBS", "WRB", "EX", "RP", "PDT", "CC", "IN"},
    }

    for wordnet_tag, tags in tag_mapping.items():
        if any(nltk_tag.startswith(tag) for tag in tags):
            return getattr(wordnet, wordnet_tag)

    return None
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize , WordNetLemmatizer , pos_tag 

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = re.sub(r"[^a-zA-Z0-9\s]+|\s+", " ", text)
    
    tokenize_words = word_tokenize(text)
    wordnet_tagged = nltk.pos_tag(tokenize_words)
    
    lemmatized_sentence = []
    for word, nltk_tag in wordnet_tagged:
        wordnet_tag = nltk_tag_to_wordnet_tag(nltk_tag)
        if wordnet_tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_word = lemmatizer.lemmatize(word, wordnet_tag)
            lemmatized_sentence.append(lemmatized_word)
    filtered_sentence = [word for word in lemmatized_sentence if word.lower() not in stop_words]
    
    return " ".join(filtered_sentence)
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
import preprocessor 

def preprocess_text(text):
    text = preprocessor.clean(text)
    return text

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer

info = pandas.read_csv("mental_health.csv", encoding="latin-1")
data = info.copy()
questions = data["text"]
answers   = data["label"]

preprocessed_questions = questions.apply(preprocess_text).apply(lemmatize_text)
vectorizer = TfidfVectorizer(encoding="latin-1" , ngram_range=(1,2) , stop_words="english")
vectorizer_Question = vectorizer.fit_transform(preprocessed_questions)

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split( vectorizer_Question , answers , test_size=0.01 , random_state=42)

classifier = SVC()
classifier.fit(x_train, y_train)


y_pred = classifier.predict(x_test)
test_score = accuracy_score(y_test, y_pred)
print("Test accuracy:", test_score)
# /////////////////////////////////////////////////////////////////////////////////////////////////////

while True:
    
    question = input("Enter a message (type 'exit' to quit):\n")
    if question.lower() == "exit":
        break
    
    question_preprocessed = preprocess_text(question)
    question_preprocessed = lemmatize_text(question_preprocessed)
    question_tfidf = vectorizer.transform([question_preprocessed])
    answer = classifier.predict(question_tfidf)
    if answer[0] == 1:
        print("The message is considered as a comment related to mental health issues.")
    elif answer[0] == 0:
        print("The message is not considered as a comment related to mental health issues.")
    else:
        print("Unable to determine the classification of the message.")
