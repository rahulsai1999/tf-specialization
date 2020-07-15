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
word_index = tokenizer.word_index
print(word_index)
```

- The tokenizer returns the list of words and the number equivalents in the form of a dictionary.
- num_words parameter means the top common words.

### Text to Sequence

- To train a neural network with text 