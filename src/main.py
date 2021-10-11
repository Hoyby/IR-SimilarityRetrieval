import random; random.seed(123)
import codecs
import string
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel
from gensim.similarities import MatrixSimilarity
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
        return list(map(lambda x: x.split(), listsOfText))

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
        return list(map(lambda word: dictionary.doc2bow(word), proccessedDocument))

    dictionary = Dictionary(proccessedDocument)
    dictionary.filter_tokens(stopWordIds(dictionary))
    bagOfWords = getBOW(dictionary)

    return bagOfWords, dictionary



def retrieval(bagOfWords, dictionary, method=0):

    def tfIdf():
        tfidf_model = TfidfModel(bagOfWords)
        tfidf_corpus  = tfidf_model[bagOfWords]
        tfidf_similarity_matrix = MatrixSimilarity(tfidf_corpus)

        return tfidf_similarity_matrix


    def lsi():
        tfidf_model = TfidfModel(bagOfWords)
        tfidf_corpus  = tfidf_model[bagOfWords]
        lsi_model = LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
        lsi_corpus = lsi_model[bagOfWords]
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






stopWords = readfile('./../stopWords.txt').split(',')
proccessedDocument, paragraphs = preProcess(readfile("./../pg3300.txt"), 'Gutenberg')
documentBOW, documentDictionary = buildDict(stopWords, proccessedDocument)

matrix = retrieval(documentBOW, documentDictionary, method=0)


queryBOW, queryDictionary = proccessQuery("What is the function of money?")

tfidf_model = TfidfModel(documentBOW)

tfidf_index = tfidf_model[queryBOW]
print(tfidf_index)

for word in tfidf_index:
    word_index = word[0]
    word_weight = word[1]
    print("index", word_index, "| word:", queryDictionary.get(word_index, word_weight), "| weight:", word_weight)


matrix_sim = MatrixSimilarity(tfidf_index)
doc2sim = enumerate(matrix_sim[tfidf_index])
top_results = sorted(doc2sim, key=lambda x: x[1], reverse=True)[:3]
# printing top 3 most relevant documents
for result in top_results:
    doc = paragraphs[result[0]]
    doc = doc.split('\n')
    print("\n[Document %d]" % result[0])
    # printing only 5 lines of the document
    for line in range(5):
        print(doc[line])