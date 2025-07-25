import numpy as np
import matplotlib.pyplot as plt

def get_sinusoidal_embeddings(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]
    print(position)
    div_term = np.exp(np.arange(0, d_model, 2) * -np.log(10000.0) / d_model)
    print(div_term)

    embeddings = np.zeros((seq_len, d_model))
    embeddings[:, 0::2] = np.sin(position * div_term)
    embeddings[:, 1::2] = np.cos(position * div_term)

    return embeddings

def plot_embeddings_for_sentence(sentence, d_model=64, dims_to_plot=8):
    # Step 1: Tokenize the sentence
    tokens = sentence.strip().split()
    seq_len = len(tokens)

    # Step 2: Get embeddings
    embeddings = get_sinusoidal_embeddings(seq_len, d_model)

    # Step 3: Plot the first N dimensions
    plt.figure(figsize=(14, 6))
    for i in range(dims_to_plot):
        plt.plot(embeddings[:, i], label=f'Dimension {i}')
    
    # Step 4: Enhance plot
    plt.xticks(ticks=np.arange(seq_len), labels=tokens, rotation=45)
    plt.title("Sinusoidal Position Embedding Curves (Token-wise)")
    plt.xlabel("Token in Sentence")
    plt.ylabel("Embedding Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Test with your own sentence
input_sentence = "The quick brown fox jumps over the lazy dog"
plot_embeddings_for_sentence(input_sentence)
