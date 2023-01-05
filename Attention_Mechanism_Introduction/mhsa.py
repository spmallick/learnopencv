from torch import nn
from einops.layers.torch import Rearrange

class MultiHeadedSelfAttention(nn.Module):
	def __init__(self, indim, adim, nheads, drop):
		'''
		indim: (int) dimension of input vector
		adim: (int) dimensionality of each attention head
		nheads: (int) number of heads in MHA layer
		drop: (float 0~1) probability of dropping a node
		
		Implements QKV MSA layer
		output = softmax(Q*K/sqrt(d))*V
		scale= 1/sqrt(d), here, d = adim
		'''
		super(MultiHeadedSelfAttention, self).__init__()
		hdim=adim*nheads
		self.scale= hdim** -0.5 #scale in softmax(Q*K*scale)*V
		self.key_lyr = self.get_qkv_layer(indim, hdim, nheads)
		#nn.Linear(indim, hdim, bias=False) 
		#there should be nheads layers
		self.query_lyr=self.get_qkv_layer(indim, hdim, nheads)
		self.value_lyr=self.get_qkv_layer(indim, hdim, nheads)
		
		self.attention_scores=nn.Softmax(dim=-1)
		self.dropout=nn.Dropout(drop)
		
		self.out_layer=nn.Sequential(Rearrange('bsize nheads indim hdim -> bsize indim (nheads hdim)'),
		nn.Linear(hdim, indim),
		nn.Dropout(drop))
	
	def get_qkv_layer(self, indim, hdim, nheads):
		'''
		returns query, key, value layer (call this function thrice to get all of q, k & v layers)
		'''
		layer=nn.Sequential(nn.Linear(indim, hdim, bias=False),
		Rearrange('bsize indim (nheads hdim) -> bsize nheads indim hdim', nheads=nheads))
		
		return layer

	def forward(self, x):
		query=self.key_lyr(x)
		key=self.query_lyr(x)
		value=self.value_lyr(x)
		
		dotp=torch.matmul(query, key.transpose(-1, -2))*self.scale
		
		scores=self.attention_scores(dotp)
		
		scores=self.dropout(scores)
		
		weighted=torch.matmul(scores, value)
		
		out=self.out_layer(weighted)
		
		return out
