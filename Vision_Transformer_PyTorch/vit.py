import torch
from torch import nn
from einops.layers.torch import Rearrange
from mhsa import MultiHeadedSelfAttention
import vitconfigs as vcfg

class TransformerEncoder(nn.Module):
	'''
	Although torch has a nn.Transformer class, it includes both encoder and decoder layers 
	(with cross attention). Since ViT requires only the encoder, we can't use nn.Transformer.
	So, we define a new class
	'''
	def __init__(self, nheads, nlayers, embed_dim, head_dim, mlp_hdim, dropout):
		'''
		nheads: (int) number of heads in MSA layer
		nlayers: (int) number of MSA layers in the transformer
		embed_dim: (int) dimension of input tokens
		head_dim: (int) dimensionality of each attention head
		mlp_hdim: (int) number of hidden dimensions in hidden layer
		dropout: (float 0~1) probability of dropping a node
		'''
		super(TransformerEncoder, self).__init__()
		self.nheads=nheads
		self.nlayers=nlayers
		self.embed_dim=embed_dim
		self.head_dim=head_dim
		self.mlp_hdim=mlp_hdim
		self.drop_prob=dropout
		
		self.salayers, self.fflayers=self.getlayers()

	def getlayers(self):
		samodules=nn.ModuleList()
		ffmodules=nn.ModuleList()
		
		for _ in range(self.nlayers):
			sam=nn.Sequential(
				nn.LayerNorm(self.embed_dim),
				MultiHeadedSelfAttention(
					self.embed_dim, 
					self.head_dim, 
					self.nheads, 
					self.drop_prob
					)
				)
			
			samodules.append(sam)

			ffm=nn.Sequential(
				nn.LayerNorm(self.embed_dim),
				nn.Linear(self.embed_dim, self.mlp_hdim),
				nn.GELU(),
				nn.Dropout(self.drop_prob),
				nn.Linear(self.mlp_hdim, self.embed_dim),
				nn.Dropout(self.drop_prob)
				)

			ffmodules.append(ffm)
		
		return samodules, ffmodules

	def forward(self, x):
		for (sal,ffl) in zip(self.salayers, self.fflayers):
			x = x+sal(x)
			x = x+ffl(x)
		
		return x

class VisionTransformer(nn.Module):
	def __init__(self, cfg):
		super(VisionTransformer, self).__init__()

		input_size=cfg['input_size']
		self.patch_size=cfg['patch_size']
		self.embed_dim=cfg['embed_dim']
		salayers=cfg['salayers']
		nheads=cfg['nheads']
		head_dim=cfg['head_dim']
		mlp_hdim=cfg['mlp_hdim']
		drop_prob=cfg['drop_prob']
		nclasses=cfg['nclasses']
		
		self.num_patches=(input_size[0]//self.patch_size)*(input_size[1]//self.patch_size) + 1

		self.patch_embedding=nn.Sequential(
			Rearrange('b c (h px) (w py) -> b (h w) (px py c)', px=self.patch_size, py=self.patch_size),
			nn.Linear(self.patch_size*self.patch_size*3, self.embed_dim)
			)

		self.dropout_layer=nn.Dropout(drop_prob)

		self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
		# similar to BERT, the cls token is introduced as a learnable parameter 
		# at the beginning of the ViT model. This token is evolved with self attention
		# and finally used to classify the image at the end. Tokens from all patches 
		# are IGNORED.

		self.positional_embedding=nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim))
		#Learnable position embedding

		self.transformer=TransformerEncoder(
			nheads=nheads, 
			nlayers=salayers, 
			embed_dim=self.embed_dim,
			head_dim=head_dim,
			mlp_hdim=mlp_hdim,
			dropout=drop_prob
			)
		
		self.prediction_head=nn.Sequential(nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, nclasses))

	def forward(self, x):
		#x is in NCHW format
		npatches=(x.size(2)//self.patch_size)*(x.size(3)//self.patch_size) + 1
		embed = self.patch_embedding(x)
		
		x=torch.cat((self.cls_token.repeat(x.size(0),1,1), embed), dim=1)
		#repeat class token for every sample in batch and cat along patch dimension, so class token is trated just like any patch
		
		if npatches==self.num_patches:
			x+=self.positional_embedding
			#this will work only if size of input image is same as that specified in the constructor
		else:
			interpolated=nn.functional.interpolate(
				self.positional_embedding[None], #insert dummy dimension
				(npatches, self.embed_dim), 
				mode='bilinear'
				) 
			#we use bilinear but only linear interp will be used
			x+=interpolated[0] #remove dummy dimension

		x=self.dropout_layer(x)
		
		x= self.transformer(x)
		
		x= x[:,0,:] 
		#use the first token for classification and ignore everything else
		
		pred=self.prediction_head(x)

		return pred

if __name__ == '__main__':
	net=VisionTransformer(vcfg.base)
	nparams=sum(p.numel() for p in net.parameters() if p.requires_grad)
	print(f'Created model with {nparams} parameters')
	x=torch.randn(1,3,224,224)
	y=net(x)
	print(y.shape)
	print('Verified Vision Transformer forward pass')