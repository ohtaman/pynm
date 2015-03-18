#!/usr/bin/env python
# -*- coding:utf-8 -*-

import logging
import random


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class LDA:
    class Entry:
        def __init__(self,
                     document_id,
                     token_id,
                     topic=None):
            self.document_id = document_id
            self.token_id = token_id
            self.topic = topic

        def __str__(self):
            return "<%s, %s, %s>" % (self.document_id,
                                     self.token_id,
                                     self.topic)

        def __repr__(self):
            return self.__str__()


    def __init__(self):
        self._entries = []
        self._document_sizes = []
        self._num_documents = 0
        self._token_ids = {}

    def add_document(self, tokens):
        document_id = self._get_document_id(tokens)
        self._document_sizes.append(len(tokens))
        logger.debug("add document. length = %s, tokens=%s"
                     % (len(tokens), tokens))
        for token in tokens:
            token_id = self._get_token_id(token)
            self._entries.append(LDA.Entry(document_id, token_id))

    def _get_document_id(self, tokens):
        self._num_documents += 1
        return self._num_documents - 1

    def _get_token_id(self, token):
        if token in self._token_ids:
            return self._token_ids[token]
        else:
            token_id = len(self._token_ids)
            self._token_ids[token] = token_id
            return token_id

    def _assign_random_topics(self, entries, num_topic):
        for entry in entries:
            entry.topic = random.randint(0, num_topics - 1)
        return entry

    def generate_topics(self, num_topics, alpha=0.5, beta=0.1, num_iter=100):
        # assign random topics
        for entry in self._entries:
            entry.topic = random.randint(0, num_topics - 1)

        topic_sizes = [0]*num_topics
        token_topic_sizes = [[0]*num_topics]*len(self._token_ids)
        document_topic_sizes = [[0]*num_topics]*self._num_documents

        for entry in self._entries:
            topic_sizes[entry.topic] += 1
            token_topic_sizes[entry.token_id][entry.topic] += 1
            document_topic_sizes[entry.document_id][entry.topic] += 1

        # gibbs sampling
        for i in range(num_iter):
            # logger.debug("iteration %s: %s" % (i, self._entries))
            for entry in self._entries:
                topic_sizes[entry.topic] -= 1
                token_topic_sizes[entry.token_id][entry.topic] -= 1
                document_topic_sizes[entry.document_id][entry.topic] -= 1
                self._document_sizes[entry.document_id] -= 1

                probs = []
                for topic in range(num_topics):
                    probs.append(
                        (alpha + document_topic_sizes[entry.document_id][topic])
                        *(beta + token_topic_sizes[entry.token_id][topic])
                        /(beta*len(self._token_ids) + topic_sizes[topic]))
                print probs
                p_ = sum(probs)*random.random()
                accum = 0
                for topic, prob in enumerate(probs):
                    accum += prob
                    if accum >= p_:
                        entry.topic = topic
                        print topic
                        break

                topic_sizes[entry.topic] += 1
                token_topic_sizes[entry.token_id][entry.topic] += 1
                document_topic_sizes[entry.document_id][entry.topic] += 1


if __name__ == '__main__':
    lda = LDA()
    lda.add_document(["a", "b", "c", "d", "a", "b"])
    lda.add_document(["a", "b", "c", "d", "a", "b"])
    lda.add_document(["a", "y", "c", "d", "1", "b"])
    lda.add_document(["a", "y", "g", "d", "d", "d"])
    lda.add_document(["y", "y", "c", "d", "v", "b"])
    lda.add_document(["r", "b", "c", "k", "a", "b"])
    lda.add_document(["a", "b", "c", "k", "v", "f"])
    lda.add_document(["1", "b", "c", "k", "w", "1"])
    lda.add_document(["c", "b", "c", "p", "9", "b"])
    lda.add_document(["1", "2", "3", "4", "5", "6"])
    lda.add_document(["1", "2", "3", "4", "7", "6"])
    lda.add_document(["1", "2", "3", "8", "7", "6"])
    lda.add_document(["1", "2", "9", "8", "7", "6"])
    lda.generate_topics(3)

    for entry in lda._entries:
        print entry
