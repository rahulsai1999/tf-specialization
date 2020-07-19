## Week 2 - Word Embeddings

- Word Embeddings are simply vector representation of words over large dimensions generated from a large corpus.
- Word Embeddings carry semantic value i.e closely related words are closer to each other in the vector space.

### IMDB dataset

- Contains 50,000 reviews labeled as positive and negative.
- Readily available in the TensorFlow dataset API.

### Building Word Embeddings

```py
import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']

training_sentences=[]
training_labels=[]
testing_sentences=[]
testing_labels=[]

for s,l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())

for s,l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())

training_labels_final=np.array(training_labels)
testing_labels_final=np.array(testing_labels)

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV"

from tensorflow.keras.preprocessing.text import Tokenizer
from.tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index=tokenizer.word_index

sequences=tokenizer.texts_to_sequences(training_sentences)
padded=pad_sequences(sequences,maxlen=max_length,truncating=trunc_type)

testing_sequences=tokenizer.texts_to_sequences(testing_sentences)
testing_padded=pad_sequences(testing_sequences,maxlen=max_length)

model= tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

# View the embeddings

e = models.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # (vocab_size,embedding_dim) or (10000,16)

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

```

### How to use the vectors?

- Similar words are given similar vectors in this multidimensional vector space, and overtime they can be clustered together.
- In the case of the IMDB dataset, the sentiments associated to each word is designated with either positive or negative.
- Closer words are associated together in the vector space.
