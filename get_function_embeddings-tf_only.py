"""This function gets the tfidf values for each function in openssl vs the list of functions in openssl for x86 architecture."""

from typing import Dict
# import cupy
import math
import numpy as np
import os
from os.path import expanduser


def asnumpy(x):
    # if cupy is not None:
    #     return cupy.asnumpy(x)
    # else:
    return np.asarray(x)


def write(f_name, matrix, file):
    m = asnumpy(matrix)
    function_name = f_name.split('/')[-1]
    print(function_name + ' ' + ' '.join(['%.6g' % x for x in m]), file=file)


# Raw count of term in a document, term frequency
def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bowCount)
    # print(tfDict)
    return tfDict


# IDFs for x86 For each instruction in a function, if that instruction exists in the other instructions of openssl increase count.
# Do not use the instruction list output of vecmap as a single file, this returns log(1/1) = 0
# Logarithmic value of: number of total documents divided by number of documents that contain the term.
def computeIDF(function, docList):
    idfDict = {}
    N = len(docList)
    idfDict = dict.fromkeys(function.keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if word in idfDict:
                if val > 0:
                    idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))  # might be (val + 1)
    return idfDict


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val * idfs[word]
    return tfidf


def computeFunctionEmbedding(tfidfBow, bow, vec_output, function_name, output_file_path):
    scalars, words, embedding_values, function_embeddings, = [], [], [], []

    # Get embeddings
    with open(vec_output, 'r') as V:
        output = V.readlines()

    # For each word and value in the tfidf results
    for word, val in tfidfBow.items():
        if word in bow:
            scalars.append(val)
            words.append(word.strip())

    # Removes the words from the vecmap embeddings. Output array of [1 by 200] values
    for line in output[1:]:
        # If the word matches a word from the function multiply embedding by scalar
        line_word = line.split(' ', 1)[0]
        if line_word in words:
            line_num = line.split(' ', 1)[1]
            line_num = [float(s) for s in line_num.split(' ')]
            embedding_values.append(line_num)

    # Multiply embeddings by scalars
    for i, value in enumerate(embedding_values):
        embedding_values[i] = np.multiply(value, scalars[i])

    # Create function embedding by a summation of the values
    function_embedding = sum(embedding_values)
    f_name = function_name.split('_formatted_', 1)[0]
    zipped = f_name, tuple(function_embedding)
    function_embeddings.append(zipped)

    # Print embeddings to file
    outfile = output_file_path + 'openssl_function_embeddings-tf_only.txt'
    output_emb = open(outfile, mode='a')
    write(f_name, function_embedding, output_emb)


def main():
    HOME = expanduser("~")
    vecmap_output = HOME + '/Desktop/Data/vecmap_output/x86.txt'
    FILE_PATH = HOME + '/Desktop/Data/extrinsic-tasks/x86/openssl_vectors/vec/'
    FILE_LIST = os.listdir(FILE_PATH)
    OUTPUT_PATH = HOME + '/Desktop/Data/extrinsic-tasks/knn/openssl_x86_vs_openfinddiff_ARM/'
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    DOCUMENT_LIST = []

    # This creates a complete list of functions, and a complete list of words in the functions for openssl x86
    for file in FILE_LIST:
        DOCUMENT_LIST.append(file)
        FILE_NAME = FILE_PATH + "%s" % file
        FUNCTION_NAME = FILE_NAME.split('.txt')[0]
        S1 = FILE_NAME
        with open(S1, 'r') as A:
            openssl_bow = A.readlines()
            with open('openssl_bow.txt', 'a') as tfile:
                for word in openssl_bow:
                    tfile.write(str(word))
                tfile.close()

    with open('openssl_bow.txt', 'r') as D:
        openssl_bow = D.readlines()

    DICTIONARY_LIST = {}
    dict_list = []

    # Creates a wordset and count for words in a function
    for i, file in enumerate(DOCUMENT_LIST):
        FILE_NAME = FILE_PATH + "%s" % file
        FUNCTION_NAME = FILE_NAME.split('.txt')[0]
        S2 = FILE_NAME
        with open(S2, 'r') as B:
            functionB_bow = B.readlines()
        with open('function_bow.txt', 'w') as tfile:
            tfile.write(str(functionB_bow))
            tfile.close()
        functionB_wordSet = set(functionB_bow)  # .union(set(openssl_bow))

        DICTIONARY_LIST[i] = dict.fromkeys(functionB_wordSet, 0)
        for word in functionB_bow:
            if word in DICTIONARY_LIST[i]:
                DICTIONARY_LIST[i][word] += 1
        dict_list.append(DICTIONARY_LIST[i])

    # For each function in openssl find the tf-idf scalar then multiply that vs each instruction embedding in the function to get the function embedding
    for file in FILE_LIST:
        FILE_NAME = FILE_PATH + "%s" % file
        FUNCTION_NAME = FILE_NAME.split('.txt')[0]
        S3 = FILE_NAME

        with open(S3, 'r') as C:
            function_bow = C.readlines()
            with open('function_bow.txt', 'w') as tfile:
                tfile.write(str(function_bow))
                tfile.close()

        function_wordSet = set(function_bow)

        function_dictionary = dict.fromkeys(function_wordSet, 0)

        for word in function_bow:
            function_dictionary[word] += 1

        function_tf = computeTF(function_dictionary, function_bow)  # function_dictionary dict has => frequency of words

        # Pass in all dictionaries from each function and current function working on.
        idfs = computeIDF(function_dictionary, [*dict_list])  # NOT USED IN FUNC_EMB CALCULATION

        function_tfidf = computeTFIDF(function_tf, idfs)  # NOT USED IN FUNC_EMB CALCULATION

        function_embedding = computeFunctionEmbedding(function_tf, function_bow, vecmap_output, FUNCTION_NAME,
                                                      OUTPUT_PATH)


if __name__ == "__main__":
    main()
