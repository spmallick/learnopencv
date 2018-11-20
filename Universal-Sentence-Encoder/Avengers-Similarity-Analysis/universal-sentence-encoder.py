import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"

print("Loading model from {}".format(module_url))
embed = hub.Module(module_url)

text_files = [f for f in os.listdir() if f.endswith(".txt")]

#characters = [i[:-4] for i in text_files]
character_lines = {}

def plot_similarity(labels, features, rotation):
    corr = np.inner(features, features)
    sns.set(font_scale=1.2)
    g = sns.heatmap(corr,\
        #xticklabels=labels,\
        #yticklabels=labels,\
        vmin=0,\
        vmax=1,\
        cmap="YlOrRd")
    #g.set_xticklabels(labels, rotation=rotation)
    g.set_title("Semantic Textual Similarity")
    #figure = g.get_figure()
    plt.tight_layout()
    plt.savefig("Avenger-semantic-similarity.png")
    plt.show()

def run_and_plot(session_, input_tensor_, messages_, labels_, encoding_tensor):
    message_embeddings_ = session_.run(encoding_tensor, feed_dict={input_tensor_: messages_})
    plot_similarity(labels_, message_embeddings_, 90)

print("Reading data from files...")

for fname in text_files:
    character = fname[:-4]
    print("Reading file for {}".format(character))
    character_line = ""
    with open(fname,'r') as g:
        for line in g.readlines():
            character_line+=line.strip()
        if character_line == "":
            continue
        character_lines[character]=character_line

# Select characters
print("================================")
print("Characters found:")
for i in range(len(character_lines.keys())):
    print("{}: {}".format(i,list(character_lines.keys())[i]))
print("================================")
print("Enter character index to be used:")
print("Enter q or Q to stop.")
flag = True
char_index = ""
final_character_lines = {}
characters = list(character_lines.keys())
while flag:
    char_index = input()
    if char_index.upper() == 'Q':
        flag=False
    else:
        char_index = int(char_index)
        final_character_lines[characters[char_index]]=character_lines[characters[char_index]]

#character_lines = final_character_lines

print("================================")
print("Characters selected:")
for i in range(len(character_lines.keys())):
    print("{}: {}".format(i,list(character_lines.keys())[i]))
print("================================")


similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_message_encodings = embed(similarity_input_placeholder)
#similarity_labels_placeholder= tf.placeholder(
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    run_and_plot(session, similarity_input_placeholder,\
            list(character_lines.values()),\
            list(character_lines.keys()),\
            similarity_message_encodings)
