## Project Overview
This project addresses the report problem: Does unlabelled data improve Twitter sentiment classification? Given a tweet, the models utilise semi-supervised learning to predict whether the sentiment of the message is positive or negative. The effectiveness of unlabelled data to the performance of various basic machine learning models is developed and analysed.

## Data Set Structure

The data is not open to the public because of school policy.

## Models

### K-nearest neighbour classifier

K-nearest neighbour classifier classifies an instance by finding the nearest k data points and conducting a weighted formula. 

### Naïve-Bayes classifiers

Naive-Bayes classifier has the naive assumption that features of an instance are conditionally independent given the class. It is a probabilistic generative model which calculates the join probability do classification.

### Logistic regression classifier

A logistic regression classifier optimizes the conditional probability directly since it is a probabilistic discriminative model. Logistic regression uses the regression approach to calculate the probability of belonging to one class, and it utilizes a decision boundary to classify.

### Multi-layer perceptron classifier
Multi-layer perceptron classifier constructs a neural network with different depths and widths. The depth of a neural network indicates the number of hidden layers it has, while the width indicates the number of neurons on each hidden layer. Multi-layer perceptron utilizes backpropagation to adjust the weight values among layers to make a better prediction in the next iteration.

## Feature Engineer

### TFIDF
The raw Tweet data (Schütze et al., 2008) was further feature engineered by the method term frequency-inverse document frequency pre-processing, which produced the data set that contains the 1000 highest TFIDF values based on the frequencies of the words.

### Embedding vectors
Also, with the pre-trained language model, Sentence Transformer (Reimers and Gurevych, 2019), the raw data after feature engineer produced the 384-dimensional embedding vectors, which can capture words with the same meaning, hence shorter the distance between similar data.

## Results

The experiment results demonstrate that the unlabelled data can improve the classifier performance. More details can be found in Report.pdf (https://github.com/Thomas-TTC/Boost-of-unlabelled-data/blob/main/Report.pdf)

## Reference
Aghababaei, S., & Makrehchi, M. (2016, November 1). Interpolative self-training approach for sentiment analysis. IEEE Xplore; IEEE. https://doi.org/10.1109/BESC.2016.7804475

Blodgett, S. L., Green, L., and O’Connor, B. (2016). Demographic dialectal variation in social media: A case study of African-American English. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1119–1130, Austin, Texas. Association for Computational Linguistics.

Madhoushi, Z., Hamdan, A. R., & Zainudin, S. (2015, July 1). Sentiment analysis techniques in recent works. IEEE Xplore; IEEE. https://doi.org/10.1109/SAI.2015.7237157
van Engelen, J. E., & Hoos, H. H. (2019). A survey on semi-supervised learning. Machine Learning, 109, 373–440. Springer Link. https://doi.org/10.1007/s10994-019-05855-6

Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Schütze, H., Manning, C. D., and Raghavan, P. (2008). Introduction to information retrieval, volume 39.
Cambridge University Press Cambridge.

Yang, B. (n.d.). Semi-supervised Learning for Sentiment Classification. In citeseerx.ist.psu.edu. Department of Computer Science, Cornell University. Retrieved May 8, 2022, from https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.403.339&rep=rep1&type=pdf
