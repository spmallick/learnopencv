import torch
import torch.nn as nn
 
 
class TinyNetworkBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
 
    def forward(self, query, context):
        query = self.norm1(query)
        context = self.norm1(context)
        attn_output, _ = self.attn(query, context, context)
        # Residual connection
        query = query + attn_output
        query = query + self.mlp(self.norm2(query))
        return query
    
class TRM(nn.Module):
    def __init__(self, embed_dim=512, n_reasoning_steps=6):
        super().__init__()
        self.net = TinyNetworkBlock(embed_dim)
        self.n = n_reasoning_steps
        self.input_projection = nn.Linear(embed_dim * 2, embed_dim)
        self.output_head = nn.Linear(embed_dim, 10) # For Sudoku (0-9 tokens)
 
    def forward(self, x, y, z):       
        for _ in range(self.n):
            combined_context = torch.cat((x, y), dim=-1)
            combined_context = self.input_projection(combined_context)  # shape: [B, L, 512]
            z = self.net(query=z, context=combined_context)
        y = self.net(query=y, context=z)     
        return y, z
    
embed_dim = 512
batch_size = 32
max_supervision_steps = 16 # N_sup from the paper
 
trm_model = TRM(embed_dim, n_reasoning_steps=6)
optimizer = torch.optim.Adam(trm_model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()
 
sequence_length = 81
question = torch.randn(batch_size, sequence_length, embed_dim)
true_answer_tokens = torch.randint(0, 10, (batch_size, sequence_length))
 
y = torch.zeros_like(question)
z = torch.zeros_like(question)

for step in range(max_supervision_steps):
    optimizer.zero_grad()
    y_embedding, z_next = trm_model(question, y, z)
    predicted_logits = trm_model.output_head(y_embedding)
    loss = loss_fn(predicted_logits.view(-1, 10), true_answer_tokens.view(-1))
    loss.backward()
    optimizer.step()
 
    print(f"Supervision Step {step+1}, Loss: {loss.item()}")
 
    y = y_embedding.detach()
    z = z_next.detach()