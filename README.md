# Hate Speech Detection with RNNs, LSTMs and GloVe Embeddings

Classifying tweets as **hate speech**, **offensive language** or **neither**. I compare three recurrent models and finish with a Gradio app for live predictions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winzepz/Hate-Speech-Detection-Using-RNN-LSTM-and-Word2Vec-GloVe-Embeddings/blob/main/2358150_DarwinPaudel_HateSpeechDetection.ipynb)

> **Content warning:** the dataset contains offensive and hateful language.

Coursework for module 6CS012 (Artificial Intelligence and Machine Learning), University of Wolverhampton at Herald College Kathmandu.

## Results

Test set: 4,953 tweets, stratified 80/20 split.

| Model | Test accuracy | Macro F1 | Hate speech F1 | Train time |
|---|---|---|---|---|
| SimpleRNN (trainable embedding) | 86.4% | **0.70** | 0.36 | 21 s |
| Stacked BiLSTM (trainable embedding) | **87.4%** | 0.69 | 0.36 | 34 s |
| Stacked BiLSTM + GloVe Twitter (50-d) | 84.0% | 0.65 | 0.33 | 20 s |

What I learned from it:

- **Accuracy isn't enough on data this skewed.** Offensive tweets make up about three quarters of the data, so a model can score high accuracy while struggling on the class that matters most. The hate speech class is only about 6% of tweets, and its F1 stayed around 0.35 for all three models.
- **The BiLSTM had the best accuracy, but the SimpleRNN had the best macro F1.** Which one is "best" depends on the metric you choose.
- **Pretrained embeddings didn't help here.** GloVe Twitter covered 85% of the vocabulary, but the 50-dimensional vectors did worse than embeddings learned from scratch on this task.

## Dataset

Hate Speech and Offensive Language dataset ([Davidson et al., 2017](https://github.com/t-davidson/hate-speech-and-offensive-language)): 24,783 labelled tweets. After cleaning and deduplication, 19,810 are used for training and 4,953 for testing.

## Approach

**Preprocessing:**
1. Strip the `RT @user:` retweet prefix.
2. Lowercase and expand contractions.
3. Remove URLs, mentions, hashtags, HTML entities and any non-letter characters.
4. Lemmatize.
5. Remove stopwords, but keep negations like *not* and *no*, since they change the meaning.

**Encoding:** a 20,000-word vocabulary fitted on the training set only. Sequences are padded to 14 tokens, the 95th percentile of tweet length. Balanced class weights correct for the imbalance.

**Models** (all use dropout and a Dense(64) layer before the softmax):
1. **SimpleRNN:** 64-d trainable embedding, then SimpleRNN(128) → SimpleRNN(64)
2. **BiLSTM:** 64-d trainable embedding, then BiLSTM(128) → BiLSTM(64) → batch norm
3. **BiLSTM + GloVe:** the same architecture, with the embedding initialised from `glove-twitter-50`

All three use Adam, early stopping and learning-rate reduction on plateau.

**Demo:** a Gradio interface. Type any text and it shows the predicted class plus the probability for each class. The models are saved in `.keras` format along with the tokenizer.

## Run it

Click the Colab badge above, upload `hatevsoffensive_language.csv` when the notebook asks for it, and run all cells.

## Built with

Python, TensorFlow/Keras, NLTK, gensim, scikit-learn, Gradio
