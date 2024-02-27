"""
COMS W4705 - Natural Language Processing - Fall 2023
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
from math import fsum, isclose

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 

        """
        #Part 1

        '''
        Procedure: First Check rule formats
        1. Iterate over the keys of self.lhs_to_rules and collect them in a set nonTerminals.
        2. Iterate over the keys of self.lhs_to_rules
        3. Within here ^ iterate over each value [a list of rules in the form (lhs, rhs, probability)]
        4. For each rule ^ check rhs set length
        5. If the length of the rhs set is 1 check if symbol is terminal i.e. symbol not in set nonTerminals
        6. If the length of the rhs set is 2 check if symbols are both nonterminals i.e. symbols are in set nonTerminals
        7. If both cases ^ above pass for all rules then return True else return False

        Now check probabilities
        1. ^^ As we are iterating over the keys to make the nonTerminals set, we can
        collect the probability sum for the respective key
        '''

        # set of non terminal symbols
        nonTerminals = set()
        for key in self.lhs_to_rules:
            nonTerminals.add(key)
            
            # collect probability sum for respective key
            keyProbSum = 0
            for rule in self.lhs_to_rules[key]:
                prob = rule[2]
                keyProbSum += prob

            # print("key: {}, keyProbSum: {}".format(key, keyProbSum))
            # check sum close to 1.0
            # print('{} sum is close to 1.0'.format(key),isclose(keyProbSum, 1.0))
            
            if not isclose(keyProbSum, 1.0):
                return False
        
        # iterate over the values
        for rules in self.lhs_to_rules.values():
            for rule in rules:
                ruleRHS = rule[1]
                ruleRHSLength = len(ruleRHS)

                # no symbols or more than two symbols
                if ruleRHSLength == 0 or ruleRHSLength > 2:
                    # print('rule1: ', rule)
                    return False
                
                if ruleRHSLength == 1:
                    # check symbol
                    if ruleRHS[0] in nonTerminals:
                        # print('rule2: ', rule)
                        return False
                
                if ruleRHSLength == 2:
                    # check symbols
                    firstSymbolIsNonTerminal = ruleRHS[0] in nonTerminals
                    secondSymbolIsNonTerminal = ruleRHS[1] in nonTerminals

                    if firstSymbolIsNonTerminal and secondSymbolIsNonTerminal:
                        continue
                    
                    else:
                        # print('rule3: ', rule)
                        return False
                    
        return True


if __name__ == "__main__":
    '''
    Then change the main section of grammar.py to read in the grammar, 
    print out a confirmation if the grammar is a valid PCFG in CNF or 
    print an error message if it is not. You should now be able to run 
    grammar.py on grammars and verify that they are well formed for the 
    CKY parser.  
    '''
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        '''
        # test to check if there are multiple non terminal symbols that generate
        # the same terminal symbol
        nonTerminals = set()
        terminals = set()
        for key in grammar.lhs_to_rules:
            nonTerminals.add(key)

        # retrieve terminal Symbols
        for rules in grammar.lhs_to_rules.values():
            for rule in rules:
                ruleRHS = rule[1]
                ruleRHSLength = len(ruleRHS)

                if ruleRHSLength == 1:
                    # check symbol
                    if ruleRHS[0] not in nonTerminals:
                        terminals.add((ruleRHS[0],))
        # for each terminal in terminals check if more
        # than one non terminal produces it

        for terminal in terminals:
            if len(grammar.rhs_to_rules[terminal]) > 1:
                print('terminal: {}, rule: {}'.format(terminal,grammar.rhs_to_rules[terminal]))
                
        #print(terminals)
        '''

        if grammar.verify_grammar():
            print('grammar is a valid PCFG in CNF')
        else:
            print('grammar is not a valid PCFG in CNF')

        print(grammar.rhs_to_rules[('FLIGHTS', 'FROM')])