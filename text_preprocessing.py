import string
from nltk.corpus import stopwords
import contractions
import re

def preprocess(data):
    # lowercase
    data = data.lower()
    # removing HTML tags and URLs
    data = re.sub(r'https?://\S+|www\.\S+', '', data)
    data = re.sub(r'<.* ? >', '', data)
    # removing punctuations
    data = re.sub('[%s]' % re.escape(string.punctuation), '', data)
    # removing numbers
    data = re.sub('[0-9]', '', data)
    # Expand Contractions
    data = ' '.join([contractions.fix(word) for word in data.split()])
    # removing stop words
    data = ' '.join([word for word in data.split() if word not in stopwords.words('english')])
    # removing extra spaces
    data = re.sub(' +', ' ', data)
    return data

