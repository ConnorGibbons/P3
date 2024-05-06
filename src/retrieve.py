# import package ...
import sys
import gzip
import json
import math

class Index: # Important: the methods assume that once the index is built, it isn't modified
    def __init__(self):
        self.index = {}
        self.documents = {}
        self.totalUniqueTokens = None
        self.totalTokenCount = None
        self.docCount = 0
    
    def add(self, document):
        self.docCount += 1
        i = 0
        self.documents[document['storyID']] = document
        for token in document['text'].split():
            if token not in self.index:
                self.index[token] = {}
            if document['storyID'] not in self.index[token]:
                self.index[token][document['storyID']] = []
            self.index[token][document['storyID']].append(i)
            i += 1
        return
    
    def getPostings(self, token): # Document IDs and positions of the token -- format: {documentID: [positions]}  (positions are 0-based)
        return self.index[token] if token in self.index else {}
    
    def getDocumentIDsForToken(self, token): # Document IDs containing the token
        return self.index[token].keys() if token in self.index else []
    
    def getTokenDocumentCount(self, token): # Number of documents containing the token
        return len(self.index[token]) if token in self.index else 0
    
    def getDocumentLength(self, document): # Number of tokens in the document
        return len(self.documents[document]['text'].split()) if document in self.documents else 0
    
    def getAverageDocumentLength(self): # Average number of tokens in documents
        return self.getTotalTokenCount() / self.getNumberOfDocuments()
    
    def getTokenFrequencyInDocument(self, token, document): # Number of times token appears in document
        return len(self.index[token][document]) if token in self.index and document in self.index[token] else 0
    
    def getTotalTokenFrequency(self, token): # Number of times token appears in all documents
        return sum([len(self.index[token][document]) for document in self.index[token]]) if token in self.index else 0
    
    def getUniqueTotalTokenCount(self): # Number of unique tokens in the index
        if self.totalUniqueTokens is None:
            self.totalUniqueTokens = len(self.index)
        return self.totalUniqueTokens
    
    def getTotalTokenCount(self): # Total number of tokens in the index
        if self.totalTokenCount is None:
            sum = 0
            for token in self.index:
                sum += self.getTotalTokenFrequency(token)
            self.totalTokenCount = sum
        return self.totalTokenCount
        
    
    def getNumberOfDocuments(self): # Number of documents in the index
        return self.docCount
    
    def getIndex(self):
        return self.index
    

    
class QueryResults:

    def __init__(self, method):
        self.results = []
        self.method = method
    
    def add(self, document, score):
        self.results.append(QueryResult(document, score))

    def getFinalResults(self, queryName = "testQuery", runTag = "cgibbons"):
        if self.method == "AND" or self.method == "OR":
            self.results.sort(key=lambda x: x.document)
        else:
            self.results.sort(key=lambda x: x.score, reverse=True)
        results = []
        rank = 1
        for result in self.results:
            results.append(f'{queryName:{15}} skip {result.document:{19}} {rank:{4}} {round(result.score,4)} {runTag+self.method}')
            rank += 1
        return results




class QueryResult:
    def __init__(self, document, score):
        self.document = document
        self.score = score

    def __str__(self):
        return f'{self.document} {self.score}'


class Query:
    def __init__(self, query):
        self.type = query[0]
        self.queryName = query[1]
        self.queryWords = []
        for word in query[2:]:
            self.queryWords.append(word.replace('\n', ''))

    def __str__(self):
        return f'{self.type} {self.queryName} {self.queryWords}'


def buildIndex(inputFile):
    index = Index()
    for document in inputFile['corpus']:
        index.add(document)
    return index

def runQueries(index, queriesFile, outputFile):
    queries = []
    results = []
    with open(queriesFile, 'r') as file:
        for line in file:
            query = line.split('\t')
            queries.append(Query(query))
    for query in queries:
        print(query.type)
        if query.type.upper() == 'AND':
            results.append(runANDQuery(index, query.queryWords))
        elif query.type.upper() == 'OR':
            results.append(runORQuery(index, query.queryWords))
        elif query.type.upper() == 'QL':
            results.append(runQLQuery(index, query.queryWords))
        elif query.type.upper() == 'BM25':
            results.append(runBM25Query(index, query.queryWords))
    with open(outputFile, 'w') as file:
        for i in range(len(queries)):
            for result in results[i].getFinalResults(queries[i].queryName):
                file.write(result + '\n')
    return

def runANDQuery(index, queryWords):
    documents = set()
    results = QueryResults("AND")
    firstIteration = True
    for word in queryWords:
        relevantDocs = index.getDocumentIDsForToken(word)
        if firstIteration:
            documents = set(relevantDocs)
            firstIteration = False
        else:
            documents = documents.intersection(relevantDocs)
    for document in documents:
        results.add(document, 1)
    return results

def runORQuery(index, queryWords):
    documents = set()
    results = QueryResults("OR")
    for word in queryWords:
        relevantDocs = index.getDocumentIDsForToken(word)
        documents = documents.union(relevantDocs)
    for document in documents:
        results.add(document, 1)
    return results

def runQLQuery(index, queryWords, u = 300):
    documents = set()
    results = QueryResults("QL")
    for word in queryWords:
        relevantDocs = index.getDocumentIDsForToken(word)
        documents = documents.union(relevantDocs)
    for document in documents:
        score = 0
        for word in queryWords:
            numerator = index.getTokenFrequencyInDocument(word, document) + u * (index.getTotalTokenFrequency(word) / index.getTotalTokenCount())
            denominator = index.getDocumentLength(document) + u
            score += math.log((numerator / denominator))
        results.add(document, score)
    return results

def runBM25Query(index, queryWords, k1 = 1.8, k2 = 5,b = 0.75):
    documents = set()
    results = QueryResults("BM25")
    for word in queryWords:
        relevantDocs = index.getDocumentIDsForToken(word)
        documents = documents.union(relevantDocs)
    for document in documents:
        bigK = k1 * ((1 - b) + b * (index.getDocumentLength(document) / index.getAverageDocumentLength()))
        score = 0
        for word in queryWords:
            q_fi = 1 # frequency of word in query
            f_i = index.getTokenFrequencyInDocument(word, document) # frequency of word in document
            n_i = index.getTokenDocumentCount(word) # number of documents containing word
            bigN = index.getNumberOfDocuments() # number of documents in index
            if(word == 'united' and document == '24323-art19'):
                print(f'{f_i} {q_fi} {bigN} {n_i} {index.getDocumentLength(document)} {index.getAverageDocumentLength()}')
            term1 = math.log((bigN - n_i + 0.5) / (n_i + 0.5))
            term2 =  ((k1 + 1) * f_i) / (bigK + f_i)
            term3 = ((k2 + 1) * q_fi) / (k2 + q_fi) 
            if(term1 != 0 and term2 != 0 and term3 != 0):
                if(word == 'united' and document == '24323-art19'):
                    print(f'{document} {word}: {term1 * term2 * term3}')
                score += (term1 * term2 * term3)
        results.add(document, score)
    return results



if __name__ == '__main__':
    # Read arguments from command line, or use sane defaults for IDE.
    argv_len = len(sys.argv)
    inputFile = sys.argv[1] if argv_len >= 2 else "sciam.json.gz"
    queriesFile = sys.argv[2] if argv_len >= 3 else "P3train.tsv"
    outputFile = sys.argv[3] if argv_len >= 4 else "P3train.trecrun"

    inputFile = gzip.open(inputFile, 'rt', encoding='utf-8')
    inputFileJSON = inputFile.read()
    inputFileDict = json.loads(inputFileJSON)
    index = buildIndex(inputFileDict)
    # print(index.getTokenFrequencyInDocument('united','24323-art19'))
    # print(index.getNumberOfDocuments())
    # print(index.getTokenDocumentCount('united'))
    # print(index.getDocumentLength('24323-art19'))
    # print(index.getAverageDocumentLength())
    if queriesFile == 'showIndex':
        exit(0)
        # Invoke your debug function here (Optional)
    elif queriesFile == 'showTerms':
        exit(0)
        # Invoke your debug function here (Optional)
    else:
        runQueries(index, queriesFile, outputFile)

    # Feel free to change anything