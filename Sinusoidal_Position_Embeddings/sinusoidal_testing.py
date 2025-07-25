import numpy as np
import matplotlib.pyplot as plt

def get_sinusoidal_embeddings(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -np.log(10000.0) / d_model)
    
    embeddings = np.zeros((seq_len, d_model))
    embeddings[:, 0::2] = np.sin(position * div_term)
    embeddings[:, 1::2] = np.cos(position * div_term)
    
    return embeddings

# Visualization
seq_len = 100
d_model = 64
embeddings = get_sinusoidal_embeddings(seq_len, d_model)

plt.figure(figsize=(14, 6))
for i in range(2):
    plt.plot(embeddings[:, i], label=f'Dimension {i}')
plt.title("Sinusoidal Position Embedding Curves")
plt.xlabel("Token Position")
plt.ylabel("Embedding Value")
plt.legend()
plt.grid(True)
plt.show()
