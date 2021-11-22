# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:36:26 2020

@author: Admin
"""

import spacy
nlp=spacy.load('en_core_web_sm')

# Getting the pipeline component
ner=nlp.get_pipe("ner")


#spaCy accepts training data as list of tuples.
#Each tuple should contain the text and a dictionary. 
#The dictionary should hold the start and end indices of the named enity in the text,
# and the category or label of the named entity.

TRAIN_DATA = [
              ("BALFOUR BEATTY FIRE AND RESCUE NW LIMITED", {"entities": [(0, 45, "Company_Name_1")]}),
              ("San Diego Supercomputer Center", {"entities": [(0, 31, "Company_Name_1")]}),
              ("Blackburn with Darwen Borough Council of Town Hall", {"entities": [(0,49, "Company_Name_1")]}),
              ("cPouta Community Cloud ser- vice", {"entities": [(0,34, "Company_Name_1")]}),
              ("Next Generation Mobility GmbH", {"entities": [(0,30, "Company_Name_1")]}),
              ("Apple Inc", {"entities": [(0,10, "Company_Name_1")]}),
              ("Arconic-Kofem Kft", {"entities": [(0,18, "Company_Name_1")]}),
              ("ACTRI Biostatistics Unit", {"entities": [(0,26, "Company_Name_1")]}),
              ("Check Point Software Technologies Ltd", {"entities": [(0,53, "Company_Name_1")]}),
              ("Transnet National Ports Authority", {"entities": [(0,46, "Company_Name_1")]}),
              ("The Small Business Company Ltd ", {"entities": [(0,36, "Company_Name_1")]}),
              ("NHS England South West", {"entities": [(0,24, "Company_Name_1")]}),
              ("Intermedia.Net, Incr", {"entities": [(0,23, "Company_Name_1")]}),
              ("McMaster Technical Inc", {"entities": [(0,24, "Company_Name_1")]})
              ]

#add these labels to the ner using ner.add_label() method of pipeline
for _, annotations in TRAIN_DATA:
  for ent in annotations.get("entities"):
    ner.add_label(ent[2])

#disable the other pipeline components through nlp.disable_pipes() method    
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

import random
from spacy.util import minibatch, compounding
from pathlib import Path

# TRAINING THE MODEL
with nlp.disable_pipes(*unaffected_pipes):

  # Training for 30 iterations
  for iteration in range(20):

    # shuufling examples  before every iteration
    random.shuffle(TRAIN_DATA)
    losses = {}
    # batch up the examples using spaCy's minibatch
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        texts, annotations = zip(*batch)
        nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
        print("Losses", losses)

doc = nlp("THIS AGREEMENT is made as a deed the 23 day of May 2011 BETWEEN: (1) BALFOUR BEATTY FIRE AND RESCUE NW LIMITED, a company incorporated in England and Wales (registered number 07403391) and having its registered office at 6th Floor, 350 Euston Road, Regent's Place, London NW1 3AX (the Contractor); and (2) BALFOUR BEATTY INVESTMENT HOLDINGS LIMITED, a company incorporated under the Companies Acts (company number 01198315) whose registered office is at 4th Floor, 130 Wilton Road, London SW1 V 1 LO (the Lender).")
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])