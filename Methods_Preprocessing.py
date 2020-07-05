'''
Text_Preprocessing
'''
import re

def preprocessor1(words):
    pre_word=re.sub(r"[^a-zA-Z]", " ", words.lower())
    return pre_word