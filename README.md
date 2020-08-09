# Using an Attention Based Model to Comparatively Assess Instagram Accounts

# Problem Statement
Panhellenic recruitment has been made completely virtual for this upcoming semester, putting serious constraints on the ability to recruit new pledge classes in a holistic manner. A big part of sorority recruitment is finding a place where you “fit in” or “feel at home” and it’s impossible to determine if you feel like you fit in with a group of girls through video chats, and vice versa. Since William and Mary Panhellenic recruitment seemingly “has” to happen this fall, sororities will have to utilize as much data about potential new members in order to make informed decisions. Social media, although criticized for being contrived, could pose a potential solution, allowing for sororities to learn more about potential new members and to construct an organic recruiting process in general.
![](recruitment_screenshot.png)

# Data
The data used is every Instagram photo and its corresponding caption posted by one of the ten panhellenic sororities at William and Mary. In order to access all of this data, I had to install the python module InstaLoader. The class Profile consists of a command that allows the user to download every image, caption, and comment associated with a given account. 

Images are preprocessed using InceptionV3, which loads pre-trained image weights for classification. The captions are tokenized, vocabulary limited, and then vectors are created based on the tokenized sequences. The vectors and image paths are used to create the dataset that is fed to the model.
In order to create training and testing groups, I used the train test split from sklearn to write a function that split the files based on their names. 

Given the timeframe of the assignment I limited the data to 3 sororities. In order to test the program, I plan on utilizing an instagram account from an active member of the last three pledge classes of the three sororities. I felt more comfortable using profiles of people I knew so I used my sorority, Chi Omega, my housemate’s, Kappa Kappa Gamma, and my best friend’s, Kappa Alpha Theta. 

# Model
[TensorFlow documentation of the model](https://www.tensorflow.org/tutorials/text/image_captioning#model)

The model can broken down into four steps
First, feature extraction using a CNN (this is accomplished during the preprocessing with InceptionV3)

image_model = tf.keras.applications.InceptionV3(include_top=False,
                                              weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

Modify shape of data
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

CNN Encoder (a single fully connected layer)

class CNN_Encoder(tf.keras.Model):
    		def __init__(self, embedding_dim):
        		super(CNN_Encoder, self).__init__()
        		self.fc = tf.keras.layers.Dense(embedding_dim)

    		def call(self, x):
        		x = self.fc(x)
        		x = tf.nn.relu(x)
        		return x

RNN Decoder (a GRU) attends over images to predict the words in each caption.


class RNN_Decoder(tf.keras.Model):
  		def __init__(self, embedding_dim, units, vocab_size):
    			super(RNN_Decoder, self).__init__()
    			self.units = units

self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    			self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
   			self.fc1 = tf.keras.layers.Dense(self.units)
    			self.fc2 = tf.keras.layers.Dense(vocab_size)

    			self.attention = BahdanauAttention(self.units)

  		def call(self, x, features, hidden):
    		# defining attention as a separate model
context_vector, attention_weights = self.attention(features, hidden)
    			x = self.embedding(x)
    			x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    			output, state = self.gru(x)
    			x = self.fc1(output)
    			x = tf.reshape(x, (-1, x.shape[2]))
    			x = self.fc2(x)
return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


