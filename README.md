# Comparing self-created model with a generative AI one for predicting IMDB review sentiments

## Summary of project
●	A comparison between a self-created single LSTM model made from scratch and an OpenAI LLM to predict the sentiment of an IMDB review as either ‘positive’ or ‘negative’.
●	The data was pre-processed by first being cleaned (non-meaningful words, html, and non-letters filtered out) to reduce unique variables. Then, it was tokenized and lemmatized before being padded into 2D numeric arrays to be fed into the model. 
●	The model was trained and tested, and the accuracy was found to be 88.54%, compared to the OpenAI LLM’s accuracy of 90.0% (without prompt engineering).

## Features, languages, and tools used
<img align="left" alt="Python" width="80px" style="padding-right:12px;" src="https://github.com/devicons/devicon/blob/v2.16.0/icons/python/python-original.svg" />
<img align="left" alt="TensorFlow" width="80px" style="padding-right:12px;" src="https://github.com/devicons/devicon/blob/v2.16.0/icons/tensorflow/tensorflow-original.svg" />
<img align="left" alt="Keras" width="80px" style="padding-right:12px;" src="https://github.com/devicons/devicon/blob/v2.16.0/icons/keras/keras-original.svg" />
<img align="left" alt="Jupyter" width="80px" style="padding-right:12px;" src="https://github.com/devicons/devicon/blob/v2.16.0/icons/jupyter/jupyter-original.svg" />
<img align="left" alt="Git" width="80px" style="padding-right:12px;" src="https://github.com/devicons/devicon/blob/v2.16.0/icons/git/git-plain.svg" />
<img align="left" alt="Github" width="80px" style="padding-right:12px;" src="https://github.com/devicons/devicon/blob/v2.16.0/icons/github/github-original.svg" />
<img align="left" alt="Pandas" width="80px" style="padding-right:12px;" src="https://github.com/devicons/devicon/blob/v2.16.0/icons/pandas/pandas-original.svg" />
<br><br><br><br><br>

## Self-made model explanation: 'Project1IMDB'
### Pre-processing
The goal of the preprocessing stage was to convert the words to numbers for the ML model and to reduce the number of 'unique' values for 
the ML model to better identify the correlations and increase overall model accuracy.
During this stage, each review was cleaned by removing the non-meaningful words, html, and non-letters, and the words were
converted to lower case. Then, they were tokenized and lemmatized so each review was a list of unique, meaningful words. These lists, once
converted to numbers (word embedding) were then padded to form identical sized 2D numeric arrays. 
### Model
After preprocessing, the arrays were split to form train and test sets before being fit onto a single LSTM model with an attached 
embedding layer. 'rmsprop' was chosen as the optimizer and 'sigmoid' was chosen as the activation function as the output choice was binary.
Checkpoints were created to keep track of the epochs based on 'val_accuracy' and the final model accuracy achieved was 88.54%. The last cell
shows that any future review can be tokenized and put into this model to generate a prediction of its sentiment. Further fine-tuning could
be done to increase its accuracy.
## OpenAI model: 'Project1IMDB_LLM'
This time, instead of using my own model, a pre-trained generative AI (OpenAI) LLM was used: 'gpt-3.5-turbo-instruct'. Calls were made to the
model using a simple prompt and the model's accuracy was evaluated. The model's accuracy improved by increasing the specificity of the 
prompt but the model still had some difficulty when a marked 'positive' review had some 'negative' language but that could possibly be
improved upon by introducing one - or few - shot inference. The final accuracy achieved with a better prompt was 95%.

## References
https://www.geeksforgeeks.org/bidirectional-rnns-in-nlp/
https://www.ibm.com/topics/recurrent-neural-networks
https://medium.com/analytics-vidhya/natural-language-processing-from-basics-to-using-rnn-and-lstm-ef6779e4ae66
https://www.tensorflow.org/guide/keras/working_with_rnns
https://medium.com/@mervebdurna/nlp-with-deep-learning-neural-networks-rnns-lstms-and-gru-3de7289bb4f8
https://openai.com/
