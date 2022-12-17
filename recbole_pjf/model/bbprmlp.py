# @Time   : 2022/3/23
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
pjfbole
"""

import numpy as np
import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class BBPRMLP(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BBPRMLP, self).__init__(config, dataset)
        self.neg_user_id = self.NEG_USER_ID = config['NEG_PREFIX'] + self.USER_ID
        self.embedding_size = config['embedding_size']
        self.LABEL_FIELD = config["LABEL_FIELD"]
        self.DIRECT_FIELD = config["DIRECT_FIELD"]

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.user_mlp_1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.user_mlp_2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.item_mlp_1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.item_mlp_2 = nn.Linear(self.embedding_size, self.embedding_size)
        
        self.loss = BPRLoss()
        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, user, item, direct):
        u_e = self.user_embedding(user)
        i_e = self.item_embedding(item)
        u_1 = self.user_mlp_1(u_e)
        u_2 = self.user_mlp_2(u_e)
        i_1 = self.item_mlp_1(i_e)
        i_2 = self.item_mlp_2(i_e)
        mul_1 = torch.mul(u_1, i_1).sum(dim=1)
        mul_2 = torch.mul(u_2, i_2).sum(dim=1)

        sum_1 = mul_1 * (direct == 1)
        sum_2 = mul_2 * (direct == 2)
        return sum_1 + sum_2

    def calculate_loss(self, interaction):
        pos_user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        # neg_user = interaction[self.NEG_USER_ID]
        direct = interaction[self.DIRECT_FIELD]

        score_pos = self.forward(pos_user, pos_item, direct)
        score_neg_i = self.forward(pos_user, neg_item, direct)

        loss = self.loss(score_pos, score_neg_i)
        return loss

    def predict(self, interaction, direct = 0):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        if direct == 0:
            direct = interaction[self.DIRECT_FIELD]
        return self.forward(user, item, direct)

