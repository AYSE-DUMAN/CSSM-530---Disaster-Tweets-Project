# -*- coding: utf-8 -*-
"""

@author: AyseDuman
"""

import pandas as pd
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# extract_named_entities.py

def extract_named_entities(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    entities = {'people': [], 'organizations': [], 'locations': []}
    current_chunk = []
    for subtree in chunked:
        if type(subtree) == Tree:
            if subtree.label() == 'GPE':
                locations = [token for token, pos in subtree.leaves()]
                entities['locations'].append(" ".join(locations))
            elif subtree.label() == 'ORGANIZATION':
                organizations = [token for token, pos in subtree.leaves()]
                entities['organizations'].append(" ".join(organizations))
            elif subtree.label() == 'PERSON':
                people = [token for token, pos in subtree.leaves()]
                entities['people'].append(" ".join(people))
                
    return entities
