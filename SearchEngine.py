import math
import os

from collections import defaultdict
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import time
start_time = time.clock()
all_filenames = []
Stemmed = {}
final_tokens = []


# Getting Word Count
def getcount(word):
    counter = 0
    word = word.lower()
    word = PorterStemmer().stem(word)
    for filename in all_filenames:
        counter = counter + Stemmed[filename].count(word)
    return counter


# Creating Tokens
def create_tokens():
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    corpus_root = 'C:\\Users\\visha\\PycharmProjects\\SearchEngine\\presidential_debates'
    for filename in os.listdir(corpus_root):
        file = open(os.path.join(corpus_root, filename), "r", encoding='UTF-8')
        doc = file.read()
        file.close()
        all_filenames.append(filename)

        # Converting the document into lower case
        doc = doc.lower()

        # Converting the document into tokens
        tokens = tokenizer.tokenize(doc)

        # Stopword Removal
        cachedstopwords = stopwords.words('english')
        doc = ' '.join([word for word in tokens if word not in cachedstopwords])

        stemmer = PorterStemmer()
        Temp = []
        for word in doc.split():
            Temp.append(stemmer.stem(word))
        for word in Temp:
            if word not in final_tokens:
                final_tokens.append(word)  # Creating a dictionary of tokens in all the documents
        Stemmed[filename] = Temp
    return Stemmed


# Calculating Term Frequency
def freq(all_filenames, Stemmed):
    tf = {}
    for filename in all_filenames:
        Temp = {}
        for word in Stemmed[filename]:
            Temp[word] = (Stemmed[filename].count(word))
        tf[filename] = Temp
    return tf


# Calculating Document Frequency
def document_frequency(all_filenames, final_tokens, tf):
    df = defaultdict(int)
    for filename in all_filenames:
        for word in final_tokens:
            if word in tf[filename]:
                if (df[word] > 0):
                    df[word] += 1
                else:
                    df[word] = 1
    return df


# Calculating Inverse Document Frequency
def inverse_document_frequency(final_tokens, df):
    idf = {}
    for word in final_tokens:
        idf[word] = math.log10(30 / df[word])
    return idf


# Calculating TF-IDF
def term_frequency__inverse_document_frequency(all_filenames, idf, tf):
    tf_idf = {}
    for filename in all_filenames:
        Temp = {}
        for word in tf[filename]:
            Temp[word] = (1 + math.log10(tf[filename].get(word))) * idf[word]
        tf_idf[filename] = Temp
    return tf_idf


# Normalizing the obtained TF_IDF values
def normalized_tf_idf(all_filenames, tf_idf):
    norm_tf_idf = {}
    for filename in all_filenames:
        Temp = {}
        denominator = 0
        for word in tf_idf[filename]:
            denominator += math.pow(tf_idf[filename].get(word), 2)

        denominator = math.sqrt(denominator)
        for word in tf_idf[filename]:
            if (denominator > 0):
                Temp[word] = tf_idf[filename].get(word) / denominator
            else:
                Temp[word] = 0
        norm_tf_idf[filename] = Temp
    return norm_tf_idf


# Function for getting the IDF of a token
def getidf(token):
    token=token.lower()
    return idf[token]


# Function for calculating Cosine Similarity between a query and a file
def querydocsim(query1, filename):
    t1 = 0
    norm_tf_query = norm_tf_idf_query(query1)
    for word in norm_tf_query:
        if word in norm_tf_idf[filename]:
            t1 += norm_tf_query.get(word) * norm_tf_idf[filename].get(word)
    return t1


# Function for Calculating Cosine Similarity between two files
def docdocsim(filename1, filename2):
    t1 = 0
    for word in norm_tf_idf[filename1]:
        if word in norm_tf_idf[filename2]:
            t1 += norm_tf_idf[filename1].get(word) * norm_tf_idf[filename2].get(word)
    return t1


# Calculating the Normalized TF-IDF values for a string
def norm_tf_idf_query(string):
    tf_query = {}
    temp = {}
    norm_tf_query = {}
    stem = []
    denominator = 0
    string=string.lower()
    stemmer = PorterStemmer()
    for word in string.split():
        stem.append(stemmer.stem(word))
    for word in stem:
        temp[word] = string.count(word)
    for word in temp:
        tf_query[word] = (1 + math.log10(temp.get(word)))
    for word in tf_query:
        denominator += math.pow(tf_query.get(word), 2)
    denominator = math.sqrt(denominator)
    for word in tf_query:
        norm_tf_query[word] = tf_query.get(word) / denominator
    return norm_tf_query


# Calculating Cosine Similarity of a Query
def query(string):
    norm_tf_query = norm_tf_idf_query(string)
    result = ''
    score = {}
    for filename in all_filenames:
        temp = {}
        t1 = 0
        for word in norm_tf_query:
            if word in norm_tf_idf[filename]:
                t1 += norm_tf_idf[filename].get(word) * norm_tf_query.get(word)
        score[filename] = t1

    for key in score.keys():
        if score.get(key) == max(score.values()):
            result = key
            break
    return (result)


create_tokens()
tf = (freq(all_filenames, Stemmed))
df = (document_frequency(all_filenames, final_tokens, tf))
idf = (inverse_document_frequency(final_tokens, df))
tf_idf = (term_frequency__inverse_document_frequency(all_filenames, idf, tf))
norm_tf_idf = normalized_tf_idf(all_filenames, tf_idf)

print(query('SECURITY conference ambassador'))
print(getcount('PRESIDENT'))
print('%.12f' % getidf('AGENDA'))
print('%.12f' % docdocsim('1960-10-21.txt', '1980-09-21.txt'))
print('%.12f' % querydocsim('particular constitutional amendment', '2000-10-03.txt'))
print("--- %s seconds ---" % (time.clock() - start_time))

