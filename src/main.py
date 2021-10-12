import codecs
import string
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel
from gensim.similarities import MatrixSimilarity
from gensim.utils import simple_preprocess
from nltk.stem.porter import PorterStemmer

# 1.0.
import random; random.seed(123)

# 1.1
def readfile(file):
    try:
        with codecs.open(file, "r", "utf-8") as f: 
            print(file, 'successfull read.')
            return f.read()
    except Exception:
        print(file, 'unsuccessfull read.')

# 1.2 - 1.6
def preProcess(filcontent, excludeWords=None):
    """Pre-proccesseses a document, filtering, tokenizing, punctuationremoval, whitespace removal, and stemming.

    Args:
        filcontent ([string]): String to be preProcessed
        excludeWords ([string], optional): String to sensor. Defaults to None.

    Returns:
        [list]: 2d list of proccessed words
        [list]: A list of filtered paragraphs
    """    

    print('Pre-processing started, please wait...')

    # 1.2
    def partitionParagraphs(filcontent): # Input String
        """Splits a string into a list of paragraphs.

        Args:
            filcontent ([string]): string to split

        Returns:
            [list]: list of paragraphs
        """        
        return filcontent.split('\r\n\r\n')

    # 1.3
    def removeDocsContainginWord(stringList, word):
        """Removes paragraphs containing a certain word.

        Args:
            stringList (list): list of paragraphs to filter
            word ([list]): word to filter by

        Returns:
            [list]: list of filtered paragraphs
        """        
        if word != None:
            return list(filter(lambda x: word.lower() not in x.lower(), stringList))
        else:
            return stringList

    # 1.4
    def tokenize(stringList):
        """Tokneizes a list of paragraphs, removing words < 2 and deaccents words

        Args:
            stringList ([list]): list to tokenize

        Returns:
            [list]: 2d list, each containing a paragraph
        """        
        return simple_preprocess(stringList, deacc=True, min_len=2)

    # 1.5
    def removePunctuation(doc):
        """Removed punctuation and whitespace from a paragraph

        Args:
            doc ([list]): list containing a string to remove punctuation and whitespaces.

        Returns:
            [list]: 2d list, each containing a paragraph with no punctuation or whitespace.
        """        
        expression = lambda x: x.translate(str.maketrans('', '', string.punctuation + '\n\r\t' ))
        return list(map(expression, doc))

    # 1.6
    def stemmer(docList):
        """Iterates through a 2d list, stemming all words

        Args:
            docList ([list]): 2d list of strings to be stemmed

        Returns:
            [type]: 2d list of stemmed words
        """        
        stemmer = PorterStemmer()
        expression = lambda word: stemmer.stem(word)
        if all(isinstance(word, list) for word in docList): # Check if all elements in the 'wordLists' param is of type list (i.e. 2d list)
            return list(list(map(expression, par)) for par in docList)
        else:
            return list(map(expression, docList))

    stringList = partitionParagraphs(filcontent)
    stringListFiltered = removeDocsContainginWord(stringList, excludeWords)
    docList = [tokenize(filteredString) for filteredString in stringListFiltered]
    docListNoPunctuation = [removePunctuation(doc) for doc in docList]
    docListStemmed = stemmer(docListNoPunctuation)
    print('Pre-processing completed')
    print()
    
    return docListStemmed, stringListFiltered

# 2.1 & 2.2
def buildDict(proccessedDocument, stopWords):
    """Removes stopwords and builds a dictionary for a collection

    Args:
        proccessedDocument ([list]): 2d list of pre-processed documents
        stopWords ([list]): list of stopwords
    
    Returns:
        [list]: A bagOfWords containing touples for word indexes, and the number of occurences.
        [corpora.dictionary]: A dictionary containing the words and indexes for words in the collection
    """    

    def stopWordIds(dictionary):
        ids = []
        for word in stopWords:
            try:
                ids.append(dictionary.token2id[word])
            except:
                pass
        return ids

    dictionary = Dictionary(proccessedDocument)
    dictionary.filter_tokens(stopWordIds(dictionary))
    bagOfWords = [dictionary.doc2bow(doc) for doc in proccessedDocument]

    return bagOfWords, dictionary

# 3.1 - 3.4
def getTfidfModel(corpus):
    return TfidfModel(corpus)

def getTfidfMatrix(corpus):
    tfidfModel = TfidfModel(corpus)
    tfidfCorpus = tfidfModel[corpus]
    return MatrixSimilarity(tfidfCorpus)

def getLsiModel(corpus, dictionary):
    tfidfModel = TfidfModel(corpus)
    tfidfCorpus = tfidfModel[corpus]
    return LsiModel(tfidfCorpus, id2word=dictionary, num_topics=100)

def getLsiMatrix(corpus, dictionary):
    lsiModel = getLsiModel(corpus, dictionary)
    lsiCorpus = lsiModel[corpus]
    return MatrixSimilarity(lsiCorpus)

# 3.5
def printfirst3LsiTopics():
    lsiDocModel = getLsiModel(documentCorpus, dictionary)
    
    print("\n[3.5 - First 3 Lsi Topics]")
    first3LsiTopics = lsiDocModel.show_topics(3)
    print(first3LsiTopics)
    '''
    [(0, '0.146*"labour" + 0.139*"price" + 0.127*"produc" + 0.126*"employ" + 0.122*"tax" + 0.121*"countri" + 0.121*"capit" + 0.118*"trade" + 0.118*"hi" + 0.114*"land"'), 
    (1, '0.233*"silver" + -0.230*"rent" + -0.215*"labour" + 0.207*"gold" + -0.194*"land" + -0.176*"employ" + -0.176*"profit" + -0.175*"capit" + -0.174*"stock" + 0.167*"coin"'), 
    (2, '0.358*"price" + -0.216*"trade" + 0.197*"labour" + 0.196*"silver" + 0.192*"quantiti" + -0.178*"coloni" + 0.148*"rent" + 0.144*"valu" + 0.130*"commod" + -0.128*"foreign"')]
    '''

# 4.1
def proccessQuery(query, dictionary):
    """Pre-proccesseses a query, filtering, tokenizing, punctuationremoval, whitespace removal, and stemming.

    Args:
        filcontent ([string]): Query to be preProcessed

    Returns:
        [list]: A bagOfWords containing touples for word indexes, and the number of occurences.
    """ 

    proccessedQuery, QueryParagraphs = preProcess(query)
    queryBOW = dictionary.doc2bow(proccessedQuery[0])
    return queryBOW

# 4.2
def printTfIdfWeights():
    tfidfDocModel = getTfidfModel(documentCorpus)
    tfidf_index = tfidfDocModel[queryCorpus]

    print("\n[4.2 - TF_IDF Weights]")
    # printing TF-DF weights of query
    for word in tfidf_index:
        word_index = word[0]
        word_weight = word[1]
        print("index", word_index, "| word:", dictionary.get(word_index, word_weight), "| weight:", word_weight)
    print('----------------------------------------------------------------------------------')

    
    '''
    index 224 | word: influenc | weight: 0.5167629963231175
    index 1578 | word: tax | weight: 0.25689531599662474
    index 2676 | word: econom | weight: 0.8166766815883432
    '''

# 4.3
def printTop3RelevantDocuments():
    tfidfDocModel = getTfidfModel(documentCorpus)
    tfidfDocMatrix = getTfidfMatrix(documentCorpus)
    tfidf_index = tfidfDocModel[queryCorpus]

    print(f"\n[4.3 - Top 3 Relevant Documents for query: '{q}']", end="")
    doc2sim = enumerate(tfidfDocMatrix[tfidf_index])
    topResults = sorted(doc2sim, key=lambda x: x[1], reverse=True)[:3]
    for result in topResults:
        print(f'\n[paragraph {result[0]}]')
        doc = documentParagraphs[result[0]]
        doc = doc.split('\n')
        for line in range(5):
            if line < len(doc):
                print(doc[line])
            if line + 1 < len(doc) and line == 4:
                print('. . .')
    print('----------------------------------------------------------------------------------')
    
    '''
    [paragraph 395]
    As men, like all other animals, naturally multiply in proportion to the
    means of their subsistence, food is always more or less in demand. It
    can always purchase or command a greater or smaller quantity of labour,
    and somebody can always be found who is willing to do something in order
    to obtain it. The quantity of labour, indeed, which it can purchase,
    . . .

    [paragraph 2052]
    Capitation Taxes.

    [paragraph 2062]
    The impossibility of taxing the people, in proportion to their revenue,
    by any capitation, seems to have given occasion to the invention of
    taxes upon consumable commodities. The state not knowing how to tax,
    directly and proportionably, the revenue of its subjects, endeavours to
    tax it indirectly by taxing their expense, which, it is supposed, will,
    . . .
    '''

# 4.4.1
def printTop3WeightedTopics():
    lsiDocModel = getLsiModel(documentCorpus, dictionary)

    print("\n[4.4.1 - Top 3 Topics with the most Significant Weights]", end="")
    lsiQuery = lsiDocModel[queryCorpus]
    topics = sorted(lsiQuery, key=lambda kv: -abs(kv[1]))[:3]
    for topic in enumerate(topics):
        t = topic[1][0]
        print("\n[Topic %d]" % t)
        print(lsiDocModel.show_topics()[t])
    print('----------------------------------------------------------------------------------')

    '''
    [Topic 3]
    (3, '-0.504*"tax" + -0.235*"rent" + 0.167*"trade" + 0.165*"employ" + -0.162*"upon" + 0.162*"labour" + 0.161*"capit" + 0.137*"quantiti" + -0.136*"hous" + 0.127*"foreign"')

    [Topic 5]
    (5, '-0.362*"tax" + -0.243*"capit" + -0.169*"foreign" + -0.142*"consumpt" + 0.131*"hi" + -0.129*"gold" + -0.128*"trade" + -0.115*"export" + -0.112*"annual" + -0.111*"duti"')

    [Topic 8]
    (8, '0.317*"tax" + 0.214*"coloni" + -0.194*"bank" + 0.172*"labour" + 0.169*"wage" + -0.166*"rent" + -0.165*"land" + -0.161*"bounti" + -0.155*"export" + -0.154*"corn"')
    '''
    
# 4.4.2
def printTop3RelevantParagraphs():
    lsiDocModel = getLsiModel(documentCorpus, dictionary)
    lsifDocMatrix = getLsiMatrix(documentCorpus, dictionary)
    lsiQuery = lsiDocModel[queryCorpus]

    print("\n[4.4.2 - Top 3 Most Relevant Paragraphs]", end="")
    lsi_doc2sim = enumerate(lsifDocMatrix[lsiQuery])
    lsiDocuments = sorted(lsi_doc2sim, key=lambda kv: -abs(kv[1]))[:3]
    for paragraph in lsiDocuments:
        doc = documentParagraphs[paragraph[0]]
        doc = doc.split('\n')
        print(f"\n[Document {paragraph[0]}]")
        for line in range(5):
            if line < len(doc):
                print(doc[line])
            if line + 1 < len(doc) and line == 4:
                print('. . .')
    print('----------------------------------------------------------------------------------')
    
    
    '''
    [Document 907]
    Whether the merchant whose capital exports the surplus produce of any
    society, be a native or a foreigner, is of very little importance. If he
    is a foreigner, the number of their productive labourers is necessarily
    less than if he had been a native, by one man only; and the value of
    their annual produce, by the profits of that one man. The sailors or
    . . .

    [Document 908]
    It is of more consequence that the capital of the manufacturer should
    reside within the country. It necessarily puts into motion a greater
    quantity of productive labour, and adds a greater value to the annual
    produce of the land and labour of the society. It may, however, be
    very useful to the country, though it should not reside within it. The
    . . .

    [Document 919]
    The foreign goods for home consumption may sometimes be purchased, not
    with the produce of domestic industry but with some other foreign goods.
    These last, however, must have been purchased, either immediately with
    the produce of domestic industry, or with something else that had been
    purchased with it; for, the case of war and conquest excepted, foreign
    . . .
    '''

# Document colletion
proccessedDocument, documentParagraphs = preProcess(readfile("../pg3300_bb.txt"), excludeWords='Gutenberg')
documentCorpus, dictionary = buildDict(proccessedDocument, readfile('../stopWords.txt').split(','))

# Queries
q1 = "How taxes influence Economics?"
q2 = "What is the function of money?"
q = q1 # query to be used
queryCorpus = proccessQuery(q, dictionary)

printfirst3LsiTopics()
printTfIdfWeights()
printTop3RelevantDocuments()
printTop3WeightedTopics()
printTop3RelevantParagraphs()