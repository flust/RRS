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


class CausCF(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(CausCF, self).__init__(config, dataset)
        self.neg_user_id = self.NEG_USER_ID = config['NEG_PREFIX'] + self.USER_ID
        self.embedding_size = config['embedding_size']
        # self.LABEL_FIELD = config["LABEL_FIELD"]
        self.DIRECT_FIELD = config["DIRECT_FIELD"]

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.exp_a_embedding = nn.Embedding(2, self.embedding_size)
        self.exp_b_embedding = nn.Embedding(2, self.embedding_size)

        self.loss = BPRLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, user, item, direct, biexposure):
        exp_a = 2 - direct
        exp_b = direct - 1
        exp_a = torch.floor((exp_a + biexposure) / 2).long()
        exp_b = torch.floor((exp_b + biexposure) / 2).long()

        u_e = self.user_embedding(user)
        i_e = self.item_embedding(item)
        exp_a = self.exp_a_embedding(exp_a)
        exp_b = self.exp_b_embedding(exp_b)
        ui = torch.mul(u_e, i_e).sum(dim=1)
        ua = torch.mul(u_e, exp_a).sum(dim=1)
        ub = torch.mul(u_e, exp_b).sum(dim=1)
        ia = torch.mul(i_e, exp_a).sum(dim=1)
        ib = torch.mul(i_e, exp_b).sum(dim=1)
        ab = torch.mul(exp_a, exp_b).sum(dim=1)
        return ui + 0.3 * ua + 0.3 * ib + 0.1 * ab
        return ui + ua + ib + ab + ub + ia

    def calculate_loss(self, interaction):
        pos_user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        neg_user = interaction[self.NEG_USER_ID]

        direct = interaction[self.DIRECT_FIELD]
        biexposure = interaction['biexposure']
        direct = 1
        biexposure = torch.zeros_like(pos_user)
        
        score_pos = self.forward(pos_user, pos_item, direct, biexposure)
        score_neg_i = self.forward(pos_user, neg_item, direct, biexposure)
        score_neg_u = self.forward(neg_user, pos_item, 2 * direct, biexposure)

        loss = self.loss(score_pos, score_neg_i) + self.loss(score_pos, score_neg_u)
        return loss

    def predict(self, interaction, direct=0):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        if direct == 0:
            direct = interaction[self.DIRECT_FIELD]

        biexposure = torch.ones_like(user)
        return self.forward(user, item, direct, biexposure)
       