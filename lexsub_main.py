#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List
from collections import defaultdict

import itertools
import string

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    '''
    Description: Takes in a lemma and pos (e.g. 'a', 'n', 'v')
    as params and returns a set of possible substitutes. 

    How to do this: Look up the lemma and pos in WordNet
    and retrieve all synsets that the lemma appears in.
    Finally, we obtain all lemmas that appear in any of these
    synsets. 

    ** Note to remove the '_' **
    '''
    possibleSubstitutes = set()

    for synset in wn.synsets(lemma, pos=pos):
       for lexeme in synset.lemmas():
            if lexeme.name().find("_") != -1:
               possibleSubstitute = lexeme.name().replace("_", " ")
               possibleSubstitutes.add(possibleSubstitute)

            elif lexeme.name() == lemma:
               continue

            else:
                possibleSubstitutes.add(lexeme.name())

    return list(possibleSubstitutes)

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    '''
    This function takes in a context object as input and
    predicts the possible synonym with the highest total
    occurence frequency (according to WordNet).

    ** Note: We must sum up the occurence counts for all
    senses of the word if the word and the target appear
    together in multiple synsets. **

    '''
    # lemma of target word
    lemma = context.lemma

    # pos
    pos = context.pos

    # retrieve synonym set
    targetWordSynsets = wn.synsets(lemma, pos=pos)

    synonymOccurences = defaultdict(int)

    for synset in targetWordSynsets:
        for lexeme in synset.lemmas():
            if lexeme.name() != context.lemma:
                synonymOccurences[lexeme.name()] += lexeme.count()
    
    predictedSynonym = max(synonymOccurences, key=synonymOccurences.get)
    predictedSynonymProcessed = predictedSynonym.replace("_", " ")
    
    return predictedSynonymProcessed

def getOverLapDictAndMaxOverLap(targetWordSynsets):
    stopWords = stopwords.words('english')
    maxOverlap = 0
    overLapDictionary = defaultdict(int)

    for synset in targetWordSynsets:
        synsetDefinitions = [[word.lower() for word in tokenize(synset.definition()) \
                              if word.lower() not in stopWords
                              ]]
        
        synsetLeftContext = tokenize(" ".join([word.lower() for word \
                            in context.left_context if word.lower() not in stopWords]))
        
        synsetRightContext = tokenize(" ".join([word.lower() for word \
                            in context.right_context if word.lower() not in stopWords]))
    
        validExamplesList = []
    
        for example in synset.examples():
            validExamplesList.append([word.lower() for word in tokenize(example) \
                             if word.lower() not in stopWords])
            
        for synsetHypernym in synset.hypernyms():
            synsetDefinitions.append([word.lower() for word in tokenize(synsetHypernym.definition()) \
                            if word.lower() not in stopWords])
            
            for example in synsetHypernym.examples():
                validExamplesList.append([word.lower() for word in tokenize(example) \
                                 if word.lower() not in stopWords])

        overLap = 0
        glossList = synsetDefinitions + validExamplesList

        for gloss in glossList:
            leftOverLap = len(set(gloss) & set(synsetLeftContext))
            rightOverLap = len(set(gloss) & set(synsetRightContext))
            overLap += leftOverLap + rightOverLap
        
        overLapDictionary[synset] = overLap
        
        maxOverlap = max(maxOverlap, overLap)

    return (overLapDictionary, maxOverlap)

def wn_simple_lesk_predictor(context : Context) -> str:
    '''
    This function uses Word Sense Disambiguation to select
    a synset for the target word.

    Returns: the most frequent synonym from that synset as
    a substitute.

    TODO: 
    1. Look at all the possible synsets that the target
    word appears in.
    2. Compute the overlap between the definition of the
    synset and the context of the target word.

    ** NOTE: I will need to remove stop words to account
    for words that don't contain any semantic meaning.

    can load list of English stopwords in NLK:
    stopWords = stopwords.words('english')

    '''

    # lemma of target word
    lemma = context.lemma

    # pos
    pos = context.pos

    # retrieve synonym set
    targetWordSynsets = wn.synsets(lemma, pos=pos)
    
    overLapDictionary, maxOverLap = getOverLapDictAndMaxOverLap(targetWordSynsets)
    
    topSynsets = [synset for (synset, overLap) in overLapDictionary.items() \
                  if overLap == maxOverLap]
    
    lexemes = []

    topSynsetOccurence = 0
    synsetDictionary = {}

    for synset in (topSynsets if topSynsets else targetWordSynsets):
        lexemes = []
        lexemeCount = 0

        for lexeme in synset.lemmas():
            if lexeme.name() == context.lemma:
                lexemeCount = lexeme.count()
                topSynsetOccurence = max(topSynsetOccurence, lexemeCount)
    
            else:
                lexemes.append(lexeme)
    
        if lexemes:
            synsetDictionary[synset] = (lexemeCount, lexemes)

    higherOccurenceSynsets = [(synset, lexeme[1]) for synset, lexeme\
        in synsetDictionary.items() if lexeme[0] == topSynsetOccurence]  
    
    if higherOccurenceSynsets:
        return max(itertools.chain(*[lexemes for _, lexemes in higherOccurenceSynsets])\
                ,key=lambda lexeme: lexeme.count()).name().replace("_", " ")
    else:
        return "smurf"
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        # print('context lemma: ', context.lemma, 'type: ', type(context.lemma))
        targetWordSynsets = get_candidates(context.lemma, context.pos)
        lemmas = []

        for synset in targetWordSynsets:
            try:
                lemmas.append((synset, self.model.similarity(context.lemma, synset.replace("_", " "))))

            except:
                continue
        
        if lemmas:
            return max(lemmas, key=lambda lemma: lemma[1])[0]
        else:
            return "smurf"


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        synsets = get_candidates(context.lemma, context.pos)
        sentence = ''
        left_context = ''

        for word in context.left_context:
            if word.isalpha():
                left_context = left_context + ' ' + word
            else:
                left_context = left_context + word
        
        sentence = left_context + ' [MASK]'

        rightContext = ''

        for word in context.right_context:
            if word.isalpha():
                rightContext = rightContext + ' ' + word
            else:
                rightContext = rightContext + word
        
        sentence += rightContext

        encodedInputTokens = self.tokenizer.encode(sentence)
        maskIndex = self.tokenizer.convert_ids_to_tokens(encodedInputTokens).index('[MASK]')
        inputMatrix = np.array(encodedInputTokens).reshape((1, -1))
        results = self.model.predict(inputMatrix, verbose=0)
        modelPredictions = results[0]

        topWordsIndices = np.argsort(modelPredictions[0][maskIndex])[::-1]
        topWords = self.tokenizer.convert_ids_to_tokens(topWordsIndices)

        for word in topWords:
            if word.replace("_", " ") in synsets:
                return word.replace("_", " ")
        
        return ""
    
    def myPredict(self, context : Context) -> str:
        synsets = get_candidates(context.lemma, context.pos)
        sentence = ''
        left_context = ''

        for word in context.left_context:
            if word.isalpha():
                left_context = left_context + ' ' + word
            else:
                left_context = left_context + word
        
        sentence = left_context + ' [MASK]'

        rightContext = ''

        for word in context.right_context:
            if word.isalpha():
                rightContext = rightContext + ' ' + word
            else:
                rightContext = rightContext + word
        
        # perhaps masking from both ends will be an improvement??
        sentence += rightContext + ' [MASK]'

        encodedInputTokens = self.tokenizer.encode(sentence)
        maskIndex = self.tokenizer.convert_ids_to_tokens(encodedInputTokens).index('[MASK]')
        inputMatrix = np.array(encodedInputTokens).reshape((1, -1))
        results = self.model.predict(inputMatrix, verbose=0)
        modelPredictions = results[0]

        topWordsIndices = np.argsort(modelPredictions[0][maskIndex])[::-1]
        topWords = self.tokenizer.convert_ids_to_tokens(topWordsIndices)

        for word in topWords:
            if word.replace("_", " ") in synsets:
                return word.replace("_", " ")
        
        return ""

    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    # W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    # predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = BertPredictor().myPredict(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))