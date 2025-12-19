import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import copy

from models.BaseModel import GeneralModel

class MGDCFBase(object):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--n_layers', type=int, default=3,
							help='Number of MGDCF layers.')
		parser.add_argument('--alpha', type=float, default=0.1,
							help='Value of alpha in MGDCF.')
		parser.add_argument('--beta', type=float, default=0.9,
							help='Value of beta in MGDCF.')
		return parser
	
	@staticmethod
	def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
		R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
		for user in train_mat:
			for item in train_mat[user]:
				R[user, item] = 1
		R = R.tolil()

		adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
		adj_mat = adj_mat.tolil()

		adj_mat[:user_count, user_count:] = R
		adj_mat[user_count:, :user_count] = R.T
		adj_mat = adj_mat.todok()

		def normalized_compute_adj(adj):
			# D^-1/2 * A * D^-1/2
			rowsum = np.array(adj.sum(1)) + 1e-10

			d_inv_sqrt = np.power(rowsum, -0.5).flatten()
			d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
			d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

			bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
			return bi_lap.tocoo()

		if selfloop_flag:
			norm_adj_mat = normalized_compute_adj(adj_mat + sp.eye(adj_mat.shape[0]))
		else:
			norm_adj_mat = normalized_compute_adj(adj_mat)

		return norm_adj_mat.tocsr()

	def _base_init(self, args, corpus):
		self.emb_size = args.emb_size
		self.n_layers = args.n_layers
		self.alpha = args.alpha
		self.beta = args.beta
		self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
		self._base_define_params()
		self.apply(self.init_weights)
	
	def _base_define_params(self):	
		self.encoder = MGDCFEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers, self.alpha, self.beta)

	def forward(self, feed_dict):
		self.check_list = []
		user, items = feed_dict['user_id'], feed_dict['item_id']
		u_embed, i_embed = self.encoder(user, items)

		prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)  # [batch_size, -1]
		u_v = u_embed.repeat(1,items.shape[1]).view(items.shape[0],items.shape[1],-1)
		i_v = i_embed
		return {'prediction': prediction.view(feed_dict['batch_size'], -1), 'u_v': u_v, 'i_v':i_v}

class MGDCF(GeneralModel, MGDCFBase):
	reader = 'BaseReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = MGDCFBase.parse_model_args(parser)
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		GeneralModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		out_dict = MGDCFBase.forward(self, feed_dict)
		return out_dict
		
	def loss(self, out_dict):
		predictions = out_dict['prediction']
		pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
		neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
		BPRloss = -(((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1)).clamp(min=1e-8,max=1-1e-8).log().mean()
		user_h = out_dict['u_v']
		item_h = out_dict['i_v']
		embedding_vars = [user_h, item_h]
		embedding_l2_losses = 0
		for var in embedding_vars:
			embedding_l2_losses += (var ** 2).sum() / 2
		loss = BPRloss + 1e-5 * embedding_l2_losses
		return loss
	

class MGDCFEncoder(nn.Module):
	def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers=3, alpha=0.1, beta=0.9):
		super(MGDCFEncoder, self).__init__()
		self.user_count = user_count
		self.item_count = item_count
		self.emb_size = emb_size
		self.layers = [emb_size] * n_layers
		self.norm_adj = norm_adj
		self.alpha = alpha
		self.beta = beta

		self.embedding_dict = self._init_model()
		self.norm_adj_matrix = norm_adj

	def _init_model(self):
		initializer = nn.init.xavier_uniform_
		embedding_dict = nn.ParameterDict({
			'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.emb_size))),
			'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.emb_size))),
		})
		return embedding_dict

	@staticmethod
	def sp_norm_adj(X, device=None):
		coo = X.tocoo()
		i = torch.LongTensor(np.array([coo.row, coo.col]))
		v = torch.from_numpy(coo.data).float()
		
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		return torch.sparse_coo_tensor(i, v, coo.shape, device=device).coalesce()

	def forward(self, users, items):
		device = next(self.embedding_dict.parameters()).device
		sparse_norm_adj = self.sp_norm_adj(self.norm_adj_matrix, device)
		
		alpha = self.alpha
		beta = self.beta
		k_num = len(self.layers)
		gamma = np.power(beta, k_num) + alpha * np.sum([np.power(beta, i) for i in range(k_num)])

		ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
		all_embeddings = [ego_embeddings]
		x_0 = torch.zeros_like(ego_embeddings) 

		for k in range(len(self.layers)):
			ego_embeddings = torch.sparse.mm(sparse_norm_adj, ego_embeddings)
			if k == 0:
				x_0 = ego_embeddings
			else:
				ego_embeddings = ego_embeddings * beta + x_0 * alpha

		ego_embeddings /= gamma
		all_embeddings = ego_embeddings

		user_embeddings_full = all_embeddings[:self.user_count, :]
		item_embeddings_full = all_embeddings[self.user_count:, :]

		user_embeddings = user_embeddings_full[users, :]
		item_embeddings = item_embeddings_full[items, :]

		return user_embeddings, item_embeddings
