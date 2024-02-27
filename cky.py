"""
COMS W4705 - Natural Language Processing - Fall 2023
Homework 2 - Parsing with Probabilistic Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg
import numpy as np


### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False

        Procedure: Implementing CKY algorithm only taking into account
        the PCFG without any of the probabilities for each one of the rules.
        ** Whenever we parse the rules in format (lhs, rhs, probability) we'll
        take a look at rule[0] and rule[1] respectively **

        ---Initialize---
        # Used numpy.zeros documentation
        https://numpy.org/doc/stable/reference/generated/numpy.zeros.html

        1. Create a 2d table of dimensions (n+1,n+1) where n is the length of the string
        i.e. len(tokens) where tokens could be in a string or list format
        2. We initialize the 2d table. ** Iterate over i = 0, ..., n - 1.
        For each of the diagonals ([i, i+1])

            1. Look at tokens[i] and save in currentWord
            2. Use the PCFG grammar rhs self.grammar.rhs_to_rules[(currentWord,)]. 
            This gives the list of rules that have produced (currentWord,).
            Save this as rulesMatched
            3. create  a set() nonTerminalsRetrieved
            4. Iterate over the rulesMatched
            5. For each rule add rule[0] to the set nonTerminalsRetrieved
            6. set the table [i, i+1] to the set nonTerminalsRetrieved
        ---Main Loop ---
        1. Iterate over the string for spans of length = 2, ..., n. Assign this as 'length'
            1. Iterate from 0 to (n-length). Assign this as 'i'
                1. Assign j = i + length. This represents the marker for 
                where the last word that the current span length covers
                2. Iterate over the possible 'k' split points. Assign this as k = i + 1, ..., j - 1
                3. Make a set() nonTerminalsConsidered
                **** NOTE WE HAVE TO TRY ALL POSSIBLE COMBINATIONS SINCE ckyTable[i,k],ckyTable[k,j COULD BE SETS***
                4. Use the PCFG grammar rhs self.grammar.rhs_to_rules[(ckyTable[i,k],ckyTable[k,j])].
                This gives a list of non terminals that could have produced (ckyTable[i,k],ckyTable[k,j]).
                Save this as rulesConsidered.
                5. Iterate over rulesConsidered 
                6. For each rule in rulesConsidered add rule[0] to the set nonTerminalsConsidered
                7. if [i,j] is empty (has a 0 in it) set the table [i, j] to be the set nonTerminalsConsidered.
                If it's not empty then take the union of the current contents in the set with the set nonTerminalsConsidered 
        Check if the start symbol (self.grammar.startsymbol) is in the set ckyTable[0,n]

        """

        #### Initialization ####
        rows, cols = (len(tokens) + 1, len(tokens) + 1)
        ckyTable =  np.zeros((rows,cols), dtype=object)
        tokens_length = len(tokens)

        for i in range(tokens_length):
            currentWord = tokens[i]
            # list of rules
            rulesMatched = self.grammar.rhs_to_rules[(currentWord,)]
            nonTerminalsRetrieved = set()
            
            for rule in rulesMatched:
                nonTerminalsRetrieved.add(rule[0])
            
            ckyTable[i][i + 1] = nonTerminalsRetrieved

        #### Main Loop ####

        for length in range(2, (tokens_length + 1)):
            #print('outer loop')
            for i in range(tokens_length - length + 1):
                j = i + length
                                
                for k in range(i+1,j):
                    #print("length: {}, i: {}, k: {}, j: {}".format(length, i, k, j))
                    nonTerminalsConsidered = set()
                    
                    # check if any cell has empty element
                    if ckyTable[i][k] == 0 or ckyTable[k][j] == 0:
                        continue
                    else:
                        # gather possible combinations of tuplets
                        validTupletCombinations = set()
                        firstSet = list(ckyTable[i][k])
                        secondSet = list(ckyTable[k][j])

                        for x in range(len(firstSet)):
                            for y in range(len(secondSet)):
                                if firstSet[x] != 0 and secondSet[y] != 0:
                                    validTupletCombinations.add((firstSet[x],secondSet[y] ))

                        #print('validTupletCombinations: ', validTupletCombinations)

                        for validTuple in validTupletCombinations:
                            rulesConsidered = self.grammar.rhs_to_rules[validTuple]
                            if rulesConsidered:
                                for rule in rulesConsidered:
                                    nonTerminalsConsidered.add(rule[0])
                        #print('nonTerminalsConsidered: ', nonTerminalsConsidered)

                        # check if currently empty
                        if ckyTable[i][j] == 0:
                            if nonTerminalsConsidered:
                                ckyTable[i][j] = nonTerminalsConsidered
                        
                        else:
                            # take the union of the sets
                            ckyTable[i][j] |= nonTerminalsConsidered
        
        lastItemIsZero = (ckyTable[0][tokens_length] == 0)

        if lastItemIsZero:
            return False
        elif self.grammar.startsymbol in ckyTable[0][tokens_length]:
            return True
        else:
            return False
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.

        Procedure for parse table:
        First, we make a dictionary data structure composed in the following way
        {[start of span, end of span][nonTerminal related][pair of tuples used to generate the nonTerminal related]}
        
        1. Initialize a defaultdict(list) and name it table
        2. Initialize a defaultdict(list) and name it probs
        ---- Initialization ----
        
        1.  begin with the terminal symbols in each part of the sentence input
            - Iterate in the inclusive range 0,..., n-1
            - Initialize a nonTerminalsConsidered = {probability:nonTerminal}
            - We need to check the possible nonTerminals that have made this curent tokens[i]
            - Use self.grammar.rhs_to_rules[(tokens[i],)] and save this as rulesConsidered
            - Initialize maxProb = 0
            - Initialize maxPointerProb = []
            - # assuming there will always be a max probability. That there is no case of equal probs
            - Iterate over each rule in the rulesConsidered list. If rule[2] > maxProb then 
            we update the value of maxProb = rule[2]. Update maxPointerProb = {rule[0]: maxProb}
           
            - Assign table[(i, i + 1)] = {maxPointerProb[0]: ((tokens[i], i, i + 1),(tokens[i], i, i + 1))}
            - Assign probs[(i, i + 1)] = {maxPointerProb[0]: maxPointerProb[1]}
        
        ---- Main Loop ----
        each entry in the table should have the most probable parse
        each entry in the probs should have the log prob of the best parse tree according to the grammar

        - Iterate inclusively over length = 2, ..., n
        - Iterate inclusively over i = 0, ..., n - length
        - Assign j = i + length
        - Iterate over k = i + 1, ..., j - 1
        - For each split point find all the nonTerminals that could have produced the rhs.
        i.e. use self.grammar.rhs_to_rules[(table[(i, k)], table[(k, j)]]
        - Initialize maxProb = 0
        - Initialize maxPointerProb = []
        - For each rule considered compute the following currentProb = log(rule[2]) * probs[(i, k)][table[(i, k)]] * table[(k, j)]
        - If currentProb > maxProb then maxProb = currentProb. Update maxPointerProb = {rule[0]: maxProb}
        - Assign table[(i, i + 1)] = {maxPointerProb[0]: ((i, k, table[(i, k)]), (k, j, table[k, j]))}
        - Assign probs[(i, i + 1)] = {maxPointerProb[0]: maxPointerProb[1]}

        **** Make sure to add seperate instances of back pointers. Only get max if you have same instance for example
        table[(0,2)['NP']] has two entries keep the higest one, otherwise add the new entry table[(0,2)['PP']]

        Procedure for probability table:

        """
        # TODO, part 3
        tokens_length = len(tokens)
        table = defaultdict(dict)
        probs = defaultdict(dict)

        # --- Initialize ---
        for i in range(tokens_length):
            rulesConsidered = self.grammar.rhs_to_rules[(tokens[i],)]
            #print("rulesConsidered: ", rulesConsidered)
            for rule in rulesConsidered:
                currentNonTerminal = rule[0]
                currentProb = math.log(rule[2])
                table[(i, i + 1)].update({currentNonTerminal: ((tokens[i], i, i + 1),(tokens[i], i, i + 1))})
                probs[(i, i + 1)].update({currentNonTerminal: currentProb})
              
        
        # --- Main Loop ---
        for length in range(2, tokens_length + 1):
            for i in range(tokens_length - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    leftSymbolList = list(table[(i, k)].keys())
                    rightSymbolList = list(table[(k, j)].keys())

                    # iterating over possible combinations of list 1 and list 2
                    
                    for x in range(len(leftSymbolList)):
                        for y in range(len(rightSymbolList)):
                            #print(leftSymbolList, rightSymbolList)
                            rules = self.grammar.rhs_to_rules[(leftSymbolList[x],rightSymbolList[y])]
                            #print("leftSymbol: {}, rightSymbol: {}, rules: {}".format(leftSymbolList[x], rightSymbolList[y], rules))
                            for rule in rules:
                                currentNonTerminal = rule[0]
                                # prob P( A -> B C) * P(B) * P(C)
                                currentProb = math.log(rule[2]) + probs[(i,k)][leftSymbolList[x]] + probs[(k, j)][rightSymbolList[y]]
                                if currentNonTerminal not in table[(i, j)]:
                                    table[(i, j)].update({currentNonTerminal: ((leftSymbolList[x],i, k),(rightSymbolList[y], k, j))})
                                    probs[(i, j)].update({currentNonTerminal: currentProb})
                                else:
                                    if currentProb > probs[(i, j)][currentNonTerminal]:
                                        table[(i, j)][currentNonTerminal] = ((leftSymbolList[x],i, k),(rightSymbolList[y], k, j))
                                        probs[(i, j)][currentNonTerminal] = currentProb
                    
                    
        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    print(chart)
    left = chart[(i, j)][nt][0]
    right = chart[(i, j)][nt][1]

    if left == right:
        return "('{}', '{}')".format(nt, left[0])
    
    leftSide = get_tree(chart, left[1], left[2], left[0])
    rightSide = get_tree(chart, right[1], right[2], right[0])

    return "('{}', {}, {})".format(nt,leftSide,rightSide)  
 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['i', 'need', 'a', 'flight', 'from', 'kansas', 'city', 'to', 'newark', 'on', 'the', 'first', 'of', 'july', '.']
        print(len(toks))
        print(parser.is_in_language(toks))
        #print(parser.is_in_language(toks3))
        table,probs = parser.parse_with_backpointers(toks)
        print(probs)

        #print(check_table_format(table))
        #print(check_probs_format(probs))
        #print(get_tree(table, 0, len(toks), grammar.startsymbol))