#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding=utf-8

__author__ = 'Dimitris'

import re
import string
from nltk.tokenize import sent_tokenize, word_tokenize


def get_words_from_file(file_path):
    text = read_file(file_path)
    words = word_tokenize(text)
    return words

def read_file(file_path):
    with open(file_path,'r') as f:
        content = ' '.join(f.readlines())
    content = content.decode('utf8')
    return fix_line(content.lower())


def fix_line (line):
    line = line.replace(u'ο»Ώ',' _s_ ')
    line = line.replace(u'?»?',' _s_ ')
    line = line.replace(u'\ufeff',' _s_ ')
    line = line.replace(u'--',' _s_ ')
    line = re.sub('-+', '', line)
    #line = replace_numbers(line)
    #line = replace_puncts(line)
    #line = replace_multiple_chars(line)
    return line

def replace_numbers(line):
    p = re.compile(r'[+-]*(\d+.\d+|\d+)')
    line = p.sub(' _num_ ', line )
    return line

def replace_puncts(line):
    exclude = set(string.punctuation) - set(["'","."])
    s = ''.join(ch for ch in line if ch not in exclude)
    return s
