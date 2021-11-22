# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:14:07 2020

@author: Admin
"""

import pandas as pd
import re

data = pd.read_excel('D:\\Machine Learning_Algoritms\\Extracting Info\\Clauses_Sample.xlsx')


def clean_text(text):
    text = re.sub('[0-9]+.\t','',str(text))
    # removing new line characters
    text = re.sub('\n ','',str(text))
    text = re.sub('\n',' ',str(text))
    # removing apostrophes
    text = re.sub("'s",'',str(text))
    # removing hyphens
    text = re.sub("-",' ',str(text))
    text = re.sub("— ",'',str(text))
    # removing quotation marks
    text = re.sub('\"','',str(text))
    # removing salutations
    text = re.sub("Mr\.",'Mr',str(text))
    text = re.sub("Mrs\.",'Mrs',str(text))
    # removing any reference to outside text
    text = re.sub("[\(\[].*?[\)\]]", "", str(text))
    return text

data['Clauses_clean'] = data['Clauses'].apply(clean_text)


def sentences(text):
    # split sentences and questions
    text = re.split('[.?]', text)
    clean_sent = []
    for sent in text:
        clean_sent.append(sent)
    return clean_sent

# sentences
data['sent'] = data['Clauses_clean'].apply(sentences)

