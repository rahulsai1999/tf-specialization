## Week 1 - Sentiment in Text

### Word based encodings

- Normal text encodings such as ASCII do not provide us the semantics of the letters and words.
- Instead, we provide encodings to the words in the particular sentence, which can be reused whenever that word occurs.

### Using the TensorFlow Text Preprocessing API

```py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = ['I love my dog','I love my cat']
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
```

- The tokenizer returns the list of words and the number equivalents in the form of a dictionary.
- num_words parameter means the top common words.

### Text to Sequence

- To train a neural network with text, we need to make dictionary based on the text corpus, which we did in the previous section.
- Next, we need to turn our sentences into a list of values based on the tokens defined in the dictionary in the previous step.
- Next we need to fix the size of input (each sentence) to be of the same length, similar to the input image size in the CNN input layer.

```py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = ['I love my dog','I love my cat']
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
print(word_index)
print(sequences)
```

- Note that the new sequences that you might have to infer on i.e. test data, has to be also encoded from the same dictionary.
- Tensorflow's APIs just skip the word, if they don't find the corresponding index for the word in the dictionary.

### More Tokenizer

- We need a large training data corpus in order to have a broad vocabulary for the neural network.
- We can also deal with unknown words or tokens with arguments in the Tensorflow Tokenizer API.

```py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = ['I love my dog','I love my cat']
tokenizer = Tokenizer(num_words=100,oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
print(word_index)
print(sequences)
```

- OOV given the index 1 and then it adds the other words alphabetically.

### Padding

```py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = ['I love my dog','I love my cat']
tokenizer = Tokenizer(num_words=100,oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences,padding='post',maxlen=8,truncating='post')
print(word_index)
print(sequences)
```

- padding parameter allows us to define if the padding has to be added in the beginning or the end.
- maxlen parameter allows us to fix the length of sentences.
- truncating parameter allows us to fix the removal of words from the start or the end if the length of sentence exceeds maxlen

### Sarcasm Detection

- Public dataset ![Link](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home)
- Contains a is_sarcastic label(0 or 1), the headline text and the link of the article in JSON format.

- Load Data Script

```py
import json
with open('sarcasm.json','r') as f:
    datastore=json.load(f)
sentences=[]
labels=[]
urls=[]

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])
```
