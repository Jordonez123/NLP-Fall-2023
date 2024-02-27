import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2023 
Programming Homework 1 - Trigram Language Models
Daniel Bauer

Name: Jordan Israel Ordonez Chaguay
Uni: jio2108
Date: 9/20/23
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """

    copied_sequence = [word for word in sequence]
    copied_sequence.insert(0, "START")
    copied_sequence.append("STOP")
    n_gram_tuples = []

    if n==1:
        for i in range(len(copied_sequence)):
            n_gram_tuples.append(tuple([copied_sequence[i]]))

    else:
        first_n_gram_list = []
        for i in range(n-1):
            first_n_gram_list.append("START")
        first_n_gram_list.append(copied_sequence[1])
        n_gram_tuples.append(tuple(first_n_gram_list))
        
        for i in range(1,len(copied_sequence)):
            last_appended_tuple = list(n_gram_tuples[-1])
            new_tuple_list = []

            for j in range(1, len(last_appended_tuple)):
                new_tuple_list.append(last_appended_tuple[j])
            
            if (i+1 > len(copied_sequence)-1):
                break
            new_tuple_list.append(copied_sequence[i+1])
            
            n_gram_tuples.append(tuple(new_tuple_list))

    return n_gram_tuples


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        
        # Retrieve total number of words
        self.total_number_words = 0
        # Retrieve total number of sentences
        self.total_number_sentences = 0
        generator = corpus_reader(corpusfile)
        for sentence in generator:
            self.total_number_sentences += 1
            for word in sentence:
                self.total_number_words += 1


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int) 

        ##Your code here
        for sentence in corpus:
            unigrams = get_ngrams(sentence, 1)
            bigrams = get_ngrams(sentence, 2)
            trigrams = get_ngrams(sentence, 3)
            
            for unigram in unigrams:
                if unigram not in self.unigramcounts:
                    self.unigramcounts[unigram] = 1
                else:
                    self.unigramcounts[unigram] += 1
            
            for bigram in bigrams:
                if bigram not in self.bigramcounts:
                    self.bigramcounts[bigram] = 1
                else:
                    self.bigramcounts[bigram] += 1

            for trigram in trigrams:
                if trigram not in self.trigramcounts:
                    self.trigramcounts[trigram] = 1
                else:
                    self.trigramcounts[trigram] += 1

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        bigram_used = (trigram[0], trigram[1])

        if ((self.trigramcounts[trigram] == 0) or (self.bigramcounts[bigram_used] == 0)):
            return self.raw_unigram_probability(trigram[0]) #replacing for the unigram probability 1/len(self.lexicon)
        
        trigram_probability = self.trigramcounts[trigram]/self.bigramcounts[bigram_used]

        return trigram_probability

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        unigram_used = (bigram[0], )

        if (self.bigramcounts[bigram] == 0):
            return 1/len(self.lexicon)
        
        bigram_probability = self.bigramcounts[bigram]/self.unigramcounts[unigram_used]
        
        return bigram_probability 
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  

        if (self.unigramcounts[unigram] == 0):
            return 1/len(self.lexicon)
        
        unigram_probability = self.unigramcounts[unigram]/self.total_number_words

        return unigram_probability

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        bigram_used = (trigram[1], trigram[2])
        unigram_used = (trigram[2])
        
        first_part = lambda1*self.raw_trigram_probability(trigram)
        second_part = lambda2*self.raw_bigram_probability(bigram_used)
        third_part = lambda3*self.raw_unigram_probability(unigram_used)

        smoothed_probability = first_part + second_part + third_part

        return smoothed_probability
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """

        trigrams_list = get_ngrams(sentence, 3)
        sentence_log_probability = 0

        for trigram in trigrams_list:
            trigram_log_probability = math.log2(self.smoothed_trigram_probability(trigram))
            sentence_log_probability += trigram_log_probability

        return sentence_log_probability

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """

        sentences_log_probability = 0
        word_count = 0

        for sentence in corpus:
            copied_sentence = [word for word in sentence]
            copied_sentence.insert(0, "START")
            copied_sentence.append("STOP")
            sentence_log_probability = self.sentence_logprob(sentence)
            sentences_log_probability += sentence_log_probability

            for word in sentence:
                word_count += 1

        l = sentences_log_probability/word_count
        perplexity = pow(2, (-1.0*l))
        
        return perplexity


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       

        # high
        for f in os.listdir(testdir1):
            total += 1
            pp_1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))

            if (pp_1 < pp_2):
                correct += 1

        # low
        for f in os.listdir(testdir2):
            total += 1
            pp_1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp_2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))

            if (pp_2 < pp_1):
                correct += 1
        
        return float((correct/total) * 100)

if __name__ == "__main__":

    # Test get_ngrams(sequence, n)
    # sequence = ["natural","language","processing"]
    # for i in range(1,4):
    #     print("{}_gram: ".format(i), get_ngrams(sequence, i))

    # Test count_ngrams(self, corpus)
    #model = TrigramModel(sys.argv[1])
    # print(model.trigramcounts[('START','START','the')])
    # print(model.bigramcounts[('START','the')])
    

    # Test total number of words in corpus]
    # print("total_number_words: ", model.total_number_words)

    # Test unigram probability
    # print("unigram probability: ", model.raw_unigram_probability(('the',)))

    # Test bigram probability
    # print("bigram probability: ", model.raw_bigram_probability(('START','the')))

    # Get the length of the lexicon
    #print("V length: ", len(model.lexicon))

    # Test trigram probability
    #print("trigram_probability: ", model.raw_trigram_probability(('as', 'much', 'as')))
    

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # #pp = model.perplexity(dev_corpus)
    # #print(pp)


    # Essay scoring experiment: 
    acc = essay_scoring_experiment(
        "./hw1_data/ets_toefl_data/train_high.txt", 
        "./hw1_data/ets_toefl_data/train_low.txt", 
        "./hw1_data/ets_toefl_data/test_high",
        "./hw1_data/ets_toefl_data/test_low"
        )
    print("accuracy of the prediction: {}%".format(acc))

