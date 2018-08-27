import numpy as np
import itertools

from unsupervised.Association.Rule import Rule


class Apriori():
    def __init__(self, min_sup=0.3, min_conf=0.81):

        self.min_sup = min_sup
        self.min_conf = min_conf
        self.freq_itemsets = None
        self.transactions = None

    def _calculate_sup(self, itemset):
        count = 0
        for transaction in self.transactions:
            if self._transaction_contains_items(transaction, itemset):
                count += 1
        sup = count / len(self.transactions)
        return sup

    def _get_frequent_itemsets(self, candidates):
        """ Prunes the candidates that are not frequent => returns list with 
        only frequent itemsets """
        frequent = []

        for itemset in candidates:
            sup = self._calculate_sup(itemset)
            if sup >= self.min_sup:
                frequent.append(itemset)
        return frequent

    def _has_infrequent_itemsets(self, candidate):
        """ True or false depending on the candidate has any
        subset with size k - 1 that is not in the frequent itemset """
        k = len(candidate)

        subsets = list(itertools.combinations(candidate, k - 1))
        for t in subsets:

            subset = list(t) if len(t) > 1 else t[0]
            if not subset in self.freq_itemsets[-1]:
                return True
        return False

    def _generate_candidates(self, freq_itemset):
        """ Joins the elements in the frequent itemset and prunes
        resulting sets if they contain subsets that have been determined
        to be infrequent. """
        candidates = []
        for itemset1 in freq_itemset:
            for itemset2 in freq_itemset:

                valid = False
                single_item = isinstance(itemset1, int)
                if single_item and itemset1 < itemset2:
                    valid = True
                elif not single_item and np.array_equal(itemset1[:-1], itemset2[:-1]) and itemset1[-1] < itemset2[-1]:
                    valid = True

                if valid:

                    if single_item:
                        candidate = [itemset1, itemset2]
                    else:
                        candidate = itemset1 + [itemset2[-1]]

                    infrequent = self._has_infrequent_itemsets(candidate)
                    if not infrequent:
                        candidates.append(candidate)
        return candidates

    def _transaction_contains_items(self, transaction, items):
        """ True or false depending on each item in the itemset is
        in the transaction """

        if isinstance(items, int):
            return items in transaction

        for item in items:
            if not item in transaction:
                return False
        return True

    def find_frequent_itemsets(self, transactions):
        """ Returns the set of frequent itemsets in the list of transactions """
        self.transactions = transactions

        unique_items = set(item for transaction in self.transactions for item in transaction)

        self.freq_itemsets = [self._get_frequent_itemsets(unique_items)]
        while (True):

            candidates = self._generate_candidates(self.freq_itemsets[-1])

            frequent_itemsets = self._get_frequent_itemsets(candidates)

            if not frequent_itemsets:
                break

            self.freq_itemsets.append(frequent_itemsets)

        frequent_itemsets = [
            itemset for sublist in self.freq_itemsets for itemset in sublist]
        return frequent_itemsets

    def _rules_from_itemset(self, initial_itemset, itemset):
        """ Recursive function which returns the rules where conf >= min_conf
        Starts with large itemset and recursively explores rules for subsets """
        rules = []
        k = len(itemset)

        subsets = list(itertools.combinations(itemset, k - 1))
        sup = self._calculate_sup(initial_itemset)
        for priori in subsets:

            priori = list(priori)
            priori_sup = self._calculate_sup(priori)

            conf = float("{0:.2f}".format(sup / priori_sup))
            if conf >= self.min_conf:

                posteriori = [itemset for itemset in initial_itemset if not itemset in priori]

                if len(priori) == 1:
                    priori = priori[0]
                if len(posteriori) == 1:
                    posteriori = posteriori[0]

                rule = Rule(
                    priori=priori,
                    posteriori=posteriori,
                    conf=conf,
                    sup=sup)
                rules.append(rule)

                if k - 1 > 1:
                    rules += self._rules_from_itemset(initial_itemset, priori)
        return rules

    def generate_rules(self, transactions):
        self.transactions = transactions
        frequent_itemsets = self.find_frequent_itemsets(transactions)

        frequent_itemsets = [itemset for itemset in frequent_itemsets if not isinstance(
            itemset, int)]
        rules = []
        for itemset in frequent_itemsets:
            rules += self._rules_from_itemset(itemset, itemset)

        return rules
