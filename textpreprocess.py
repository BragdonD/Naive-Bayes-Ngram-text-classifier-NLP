import re
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords

import spacy

import string
PUNCTUATION = string.punctuation


class TextPreProcessor:
  def __init__(self, language="english", lemmatization = False, stopword = False, stemmatization = False, lower = False, ponct = False, emoji = False, symbols = False, numbers = False):
    if (lemmatization == True & stemmatization == True):
      raise Exception("Can not lemmatize and stem sentences at the same time.")

    self.lemmatization = lemmatization
    self.stemmatization = stemmatization
    self.lower = lower
    self.ponct = ponct
    self.emoji = emoji
    self.stopword = stopword
    self.symbols = symbols
    self.numbers = numbers
    self.lemmatizer = spacy.load('en_core_web_sm')
    self.stemmer = nltk.SnowballStemmer("english")
    self.REPLACE_BY_SPACE_RE = re.compile('[-+/(){}\[\]\|@,;]')
    self.BAD_SYMBOLS_RE = re.compile('[0-9] {,1}')
    self.STOPWORDS = set(stopwords.words('english'))

  def cleanText(self, text):
    if text == "":
      return ""

    def lower_case(text):
      return text.lower()

    def remove_punctuation(text):
      return text.translate(str.maketrans('', '', PUNCTUATION))

    def remove_symbols(dataframe):
      return self.REPLACE_BY_SPACE_RE.sub(' ', text)

    def remove_numbers(text):
      return self.BAD_SYMBOLS_RE.sub(' ', text)

    def remove_emoji(string):
      emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
      return emoji_pattern.sub(' ', string)

    def remove_stopwords(text):
      return " ".join([word for word in str(text).split() if word not in self.STOPWORDS])

    def lemmatize(text):
      tokens = []
      for token in self.lemmatizer(text):
        tokens.append(token.lemma_)
      return " ".join(tokens)

    def stemmatize(text):
      tokens = []
      for token in text.split(" "):
        tokens.append(self.stemmer.stem(token))
      return " ".join(tokens)
        
    if(self.lower == True):
      text = lower_case(text)
    if(self.numbers == True):
      text = remove_numbers(text)
    if(self.ponct == True):
      text = remove_punctuation(text)
    if(self.symbols == True):
      text = remove_symbols(text)
    if(self.emoji == True):
      text = remove_emoji(text)
    if(self.stopword == True):
      text = remove_stopwords(text)
    if(self.lemmatization == True):
      text = lemmatize(text)
    if(self.stemmatization == True):
      text = stemmatize(text)
    
    return text