import csv
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import accuracy_score
import numpy as np
import os
from transformers import BertTokenizer, BertModel
import torch

FILE_PATH = os.path.join(os.getcwd(), 'data/reviews_mixed.csv')
N_CLUSTERS = 2
RANDOM_SEED = 5
MAX_FEATURES = 50

def read_csv(file_path):
    data = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line_count, row in enumerate(csv_reader):
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
    return data

def split_data(data):
    inputs = [data[i][0] for i in range(len(data))][:100]
    outputs = [data[i][1] for i in range(len(data))][:100]
    labelNames = list(set(outputs))

    np.random.seed(RANDOM_SEED)
    noSamples = len(inputs)
    indexes = [i for i in range(noSamples)]
    trainSample = np.random.choice(indexes, int(0.8 * noSamples), replace=False)
    testSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    return trainInputs, trainOutputs, testInputs, testOutputs, labelNames

def featureComputation(vectorizer, data):
    features = vectorizer.fit_transform(data).toarray()
    return features

def bertEmbeddings(trainingInput):
    allEmbeddings = []
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", force_download=True)
    model = BertModel.from_pretrained('bert-base-uncased', force_download=True)

    for input in trainingInput:
        input_ids = tokenizer.encode(input, add_special_tokens=True, max_length=128, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)
            contextualEmbeddings = outputs.last_hidden_state

        allEmbeddings.append(contextualEmbeddings)
    return allEmbeddings

def extract_features(text):
    word_count = len(word_tokenize(text))
    char_count = len(text)
    sentence_count = len(sent_tokenize(text))
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    adjective_count = len([word for word, tag in pos_tags if tag.startswith('JJ')])
    verb_count = len([word for word, tag in pos_tags if tag.startswith('VB')])
    return [word_count, char_count, sentence_count, adjective_count, verb_count]

def average_word_length(text):
    words = text.split()
    if len(words) == 0:
        return 0
    total_length = sum(len(word) for word in words)
    return total_length / len(words)

def main():
    data = read_csv(FILE_PATH)
    trainInputs, trainOutputs, testInputs, testOutputs, labelNames = split_data(data)

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    trainFeatures = featureComputation(vectorizer, trainInputs)
    testFeatures = featureComputation(vectorizer, testInputs)
    trainEmbeddings = bertEmbeddings(trainInputs)
    for i, embedding in enumerate(trainEmbeddings):
        print(f"Embedding for input {i + 1} has shape {embedding.shape}")

    unsupervisedClassifier = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED)
    unsupervisedClassifier.fit(trainFeatures)

    computedTestIndexes = unsupervisedClassifier.predict(testFeatures)
    computedTestOutputs = [labelNames[value] for value in computedTestIndexes]
    for i in range(0, len(testInputs)): print(testInputs[i] , "is",computedTestOutputs[i])

    print(accuracy_score(testOutputs, computedTestOutputs))

    trainAvgWordLength = np.array([average_word_length(text) for text in trainInputs]).reshape(-1, 1)
    testAvgWordLength = np.array([average_word_length(text) for text in testInputs]).reshape(-1, 1)

    trainFeaturesExtended = np.hstack((trainFeatures, trainAvgWordLength , np.array([extract_features(text) for text in trainInputs])))
    testFeaturesExtended = np.hstack((testFeatures,testAvgWordLength, np.array([extract_features(text) for text in testInputs])))
    unsupervisedClassifier.fit(trainFeaturesExtended)

    computedTestIndexes = unsupervisedClassifier.predict(testFeaturesExtended)
    computedTestOutputs = [labelNames[value] for value in computedTestIndexes]

    print(accuracy_score(testOutputs, computedTestOutputs))

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    computedTestIndexes = dbscan.fit_predict(testFeatures)
    computedTestOutputs = [labelNames[value] for value in computedTestIndexes]
    print("DBSCAN: ", accuracy_score(testOutputs, computedTestOutputs))

    agglo = AgglomerativeClustering(n_clusters=N_CLUSTERS)
    computedTestIndexes = agglo.fit_predict(testFeatures)
    computedTestOutputs = [labelNames[value] for value in computedTestIndexes]
    print("Agglomerative clustering: ", accuracy_score(testOutputs, computedTestOutputs))

if __name__ == "__main__":
    main()



