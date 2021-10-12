import random; random.seed(123)
import codecs
import string
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel
from gensim.similarities import MatrixSimilarity
from gensim.utils import simple_preprocess
from nltk.stem.porter import PorterStemmer

def readfile(file):
        with codecs.open(file, "r", "utf-8") as f: 
            return f.read()

def preProcess(fileContent, excludeWords=None):

    def partitionParagraphs(text):
        return text.split('\r\n\r\n')

    def removeParagraphsContainginWord(listsOfText, word):
        if word != None:
            return list(filter(lambda x: word.lower() not in x.lower(), listsOfText))
        else:
            return listsOfText

    def tokenize(listsOfText):
        return [simple_preprocess(doc) for doc in listsOfText]

    def removePunctuation(wordLists):
        '''
        Removes punctuation and whitespace from a list of words, or a list of list of words.
        '''
        expression = lambda x: x.translate(str.maketrans('', '', string.punctuation + '\n\r\t' ))
        if all(isinstance(x, list) for x in wordLists): # Check if all elements in the 'wordLists' param is of type list (i.e. 2d list)
            return list(list(map(expression, par)) for par in wordLists)
        else:
            return list(map(expression, wordLists))


    def stemmer(wordLists):
        '''
        Stems words in a list of words, or a list of list of words.
        '''
        stemmer = PorterStemmer()
        expression = lambda word: stemmer.stem(word.lower())
        if all(isinstance(word, list) for word in wordLists): # Check if all elements in the 'wordLists' param is of type list (i.e. 2d list)
            return list(list(map(expression, par)) for par in wordLists)
        else:
            return list(map(expression, wordLists))

    paragraphs = partitionParagraphs(fileContent)
    noHeaderFooter = removeParagraphsContainginWord(paragraphs, excludeWords)
    tokenized = tokenize(noHeaderFooter)
    noPunctuation = removePunctuation(tokenized)
    stemmd = stemmer(noPunctuation)

    return stemmd, paragraphs



def buildDict(stopWords, proccessedDocument):

    def stopWordIds(dictionary):
        ids = []
        for word in stopWords:
            try:
                ids.append(dictionary.token2id[word])
            except:
                pass
        return ids

    def getBOW(dictionary):
        return [dictionary.doc2bow(doc, allow_update=True) for doc in proccessedDocument]

    dictionary = Dictionary(proccessedDocument)
    dictionary.filter_tokens(stopWordIds(dictionary))
    bagOfWords = getBOW(dictionary)

    return bagOfWords, dictionary



def retrieval(corpus, dictionary, method=0):
    '''
    params:
        method: 0=TFIDF (default), 1=LSI
    '''

    def tfIdf(corpus):
        tfidf_model = TfidfModel(corpus)
        tfidf_corpus  = tfidf_model[corpus]
        tfidf_similarity_matrix = MatrixSimilarity(tfidf_corpus)

        return tfidf_similarity_matrix


    def lsi(corpus):
        tfidf_model = TfidfModel(corpus)
        tfidf_corpus  = tfidf_model[corpus]
        lsi_model = LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
        lsi_corpus = lsi_model[corpus]
        lsi_similarity_matrix = MatrixSimilarity(lsi_corpus)

        return lsi_similarity_matrix

    if method == 0:
        return tfIdf()

    elif method == 1:
        return lsi()


def proccessQuery(query):

    def getBOW(dictionary):
        return list(map(lambda word: dictionary.doc2bow(word), preProccessedQuery))

    preProccessedQuery, paragraphs = preProcess(query)
    dictionary = Dictionary(preProccessedQuery)
    bagOfWords = getBOW(dictionary)

    return bagOfWords, dictionary

def gettfidfmodel(BOW):
    return TfidfModel(BOW)


def printOccurences(BOW, dictionary):
    for word in BOW:
        print([[dictionary[id], freq] for id, freq in word])

def printtfidfWeights(corpus, dictionary):
    tfidf = TfidfModel(corpus)

    for doc in tfidf[corpus]:
        print([[dictionary[id], round(freq,2)] for id, freq in doc])

def printTop3Documents(tfidf_index, tfidf_query):
    for tfidf_index, similarity in sorted(enumerate(tfidf_index[tfidf_query]), key=lambda kv: -kv[1])[:3]:
        paragraph = paragraphs[tfidf_index].split("\n")
        number = tfidf_index + 1
        print("[paragraph: " + str(number) + "]")
        for i in range(5):
            print(paragraph[i])
            if (i+1) == len(paragraph):
                break
        print("\n")


# Pre Process Collection
proccessedDocument, paragraphs = preProcess(readfile("./../pg3300.txt"), 'Gutenberg')
documentBOW, documentDictionary = buildDict(readfile('./../stopWords.txt').split(','), proccessedDocument)

# Pre Process query
queryBOW, queryDictionary = proccessQuery("How taxes influence Economics?")

printOccurences(queryBOW, queryDictionary)
printtfidfWeights(queryBOW, queryDictionary)
tfidf = TfidfModel(corpus)

tfidf_index = retrieval(documentBOW, documentDictionary, method=0)
printTop3Documents(tfidf_index)


# matrix_sim = MatrixSimilarity(tfidf_index)
# doc2sim = enumerate(matrix_sim[tfidf_index])
# top_results = sorted(doc2sim, key=lambda x: x[1], reverse=True)[:3]
# # printing top 3 most relevant documents
# for result in top_results:
#     doc = paragraphs[result[0]]
#     doc = doc.split('\n')
#     print("\n[Document %d]" % result[0])
#     # printing only 5 lines of the document
#     for line in range(5):
#         print(doc[line])