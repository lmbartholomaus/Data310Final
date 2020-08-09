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


