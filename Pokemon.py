# Importing necessary libraries
import pandas as pd
import numpy as np

# Preprocessing the Data

data = pd.read_csv('Pokemon.csv')
data.fillna(value = 'NA', axis = 0, inplace = True)
data.drop(['#', 'Name', 'Total', 'Generation'], axis = 1, inplace = True)
def func(x):
    if x == True:
        return 1
    else:
        return 0
data['Legendary'] = data['Legendary'].apply(lambda x: func(x))
data.rename(columns = {'Sp Atk': 'Sp.Atk', 'Sp Def': 'Sp.Def', 'Type 1': 'Type.1', 'Type 2':'Type.2'}, inplace = True)


# Importing DNN Estimator API from Tensorflow

import tensorflow as tf

# Creating Categorical Feature Columns

t1 = tf.feature_column.categorical_column_with_vocabulary_list('Type.1', 
	['Grass', 'Fire', 'Water', 'Bug', 'Normal', 'Poison', 'Electric',
     'Ground', 'Fairy', 'Fighting', 'Psychic', 'Rock', 'Ghost', 'Ice',
     'Dragon', 'Dark', 'Steel', 'Flying'])
t2 = tf.feature_column.categorical_column_with_vocabulary_list('Type.2', 
	['Poison', 'NA', 'Flying', 'Dragon', 'Ground', 'Fairy', 'Grass',
     'Fighting', 'Psychic', 'Steel', 'Ice', 'Rock', 'Dark', 'Water',
     'Electric', 'Fire', 'Ghost', 'Bug', 'Normal'])

# Creating Numerical Feature Columns

hp = tf.feature_column.numeric_column('HP', shape = (1,))
atk = tf.feature_column.numeric_column('Attack', shape = (1,))
df = tf.feature_column.numeric_column('Defense', shape = (1,))
satk = tf.feature_column.numeric_column('Sp.Atk', shape = (1,))
sdf = tf.feature_column.numeric_column('Sp.Def', shape = (1,))
spd = tf.feature_column.numeric_column('Speed', shape = (1,))

# Embedding Categorical Feature Columns in _Dense Columns

t11 = tf.feature_column.embedding_column(t1, 18)
t12 = tf.feature_column.embedding_column(t2, 19)

# Listing all Feature columns

feat_cols = [t11, t12, hp, atk, df, satk, sdf, spd]

# Training

X = data.drop(['Legendary'], axis = 1)
y = data['Legendary']
train_input_fn = tf.estimator.inputs.pandas_input_fn(X, y, batch_size=25, shuffle=True, num_epochs=1000)
model = tf.estimator.DNNClassifier([5, 5, 5], feat_cols, n_classes = 2)
model.train(train_input_fn)

# Predictions

print("Hello there Poke-Trainer, please enter the following information about your Pokemon when asked!")
print("What is its type?")
type1 = input()
print("What is its other type? If there is no other type, kindly enter 'NA'.")
type2 = input()
print("What is its maximum HP?")
hp_ = int(input())
print("What is its maximum basic attack?")
atk_ = int(input())
print("What is its maximum basic defense?")
df_ = int(input())
print("What is its maximum special attack?")
satk_ = int(input())
print("What is its maximum special defense?")
sdf_ = int(input())
print("What is its maximum speed?")
spd_ = int(input())

datf = pd.DataFrame([[type1, type2, hp_, atk_, df_, satk_, sdf_, spd_]], columns = ['Type.1', 'Type.2', 'HP', 
	'Attack', 'Defense', 'Sp.Atk', 'Sp.Def', 'Speed'])

predict_input_fn = tf.estimator.inputs.pandas_input_fn(df, shuffle = False)
a = model.predict(predict_input_fn)
k = list(a)
if k[0]['class_ids'][0] == 0:
	print("Your Pokemon is not a Legendary.")
else:
	print("Whoa! Your Pokemon is a Legendary.")
