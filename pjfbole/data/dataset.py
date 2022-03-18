# @Time   : 2022/03/01
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
pjfbole.data.pjf_dataset
##########################
"""

import os
import numpy as np
import pandas as pd
import torch

from recbole.data.dataset import Dataset
from recbole.utils import set_color
from recbole.data.interaction import Interaction


class PJFDataset(Dataset):
    """:class:`PJFDataset` is inherited from :class:`recbole.data.dataset.Dataset`

    """
    def __init__(self, config):
        super().__init__(config)

    def change_direction(self):
        """Change direction for Validation and testing.
        """
        self.uid_field, self.iid_field = self.iid_field, self.uid_field
        self.user_feat, self.item_feat = self.item_feat, self.user_feat
        self.user_sents, self.item_sents = self.item_sents, self.user_sents

    def _change_feat_format(self):
        super(PJFDataset, self)._change_feat_format()
        self._sents_processing()

    def _sents_processing(self):
        def fill_nan(value):
            if isinstance(value, np.ndarray):
                return value
            else:
                return np.zeros([self.config['max_sent_num'], self.config['max_sent_len']])

        if self.user_sents is not None:
            new_usents_df = pd.DataFrame({self.uid_field: np.arange(self.user_num)})
            self.user_sents = pd.merge(new_usents_df, self.user_sents, on=self.uid_field, how='left')
            self.user_sents[self.usents_field].fillna(value=0, inplace=True)
            self.user_sents[self.usents_field] = \
                self.user_sents[self.usents_field].apply(fill_nan)

        if self.item_sents is not None:
            new_isents_df = pd.DataFrame({self.iid_field: np.arange(self.item_num)})
            self.item_sents = pd.merge(new_isents_df, self.item_sents, on=self.iid_field, how='left')
            self.item_sents[self.isents_field].fillna(value=0, inplace=True)
            self.item_sents[self.isents_field] = \
                self.item_sents[self.isents_field].apply(fill_nan)

        self.user_sents = self._sents_dataframe_to_interaction(self.user_sents)
        self.item_sents = self._sents_dataframe_to_interaction(self.item_sents)

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Returns:
            list: List of built :class:`Dataset`.
        """
        datasets = super(PJFDataset, self).build()

        if self.config['multi_direction']:
            direct_field = self.config['DIRECT_FIELD']
            geek_direct = datasets[0].field2token_id[direct_field]['0']
            valid_g = self.copy(datasets[1].inter_feat[datasets[1].inter_feat[direct_field] == geek_direct])

            valid_j = self.copy(datasets[1].inter_feat[datasets[1].inter_feat[direct_field] != geek_direct])
            valid_j.change_direction()

            test_g = self.copy(datasets[2].inter_feat[datasets[2].inter_feat[direct_field] == geek_direct])

            test_j = self.copy(datasets[2].inter_feat[datasets[2].inter_feat[direct_field] != geek_direct])
            test_j.change_direction()
            return [datasets[0], valid_g, valid_j, test_g, test_j]
        return datasets

    def _get_field_from_config(self):
        """Initialization common field names.
        """
        super(PJFDataset, self)._get_field_from_config()
        self.usents_field = self.config['USER_SENTS_FIELD']
        self.isents_field = self.config['ITEM_SENTS_FIELD']
        self.logger.debug(set_color('usents_field', 'blue') + f': {self.usents_field}')
        self.logger.debug(set_color('isents_field', 'blue') + f': {self.isents_field}')

    def _load_data(self, token, dataset_path):
        """Load features of the resume and job description.

        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
        """
        super(PJFDataset, self)._load_data(token, dataset_path)
        self.user_sents = self._load_user_or_item_sents(token, dataset_path, 'usents', 'uid_field', 'usents_field')
        self.item_sents = self._load_user_or_item_sents(token, dataset_path, 'isents', 'iid_field', 'isents_field')
        self.filter_data_with_no_sents()

    def _load_user_or_item_sents(self, token, dataset_path, suf, field_name, sents_field_name):
        """Load user/item sents.
        Returns:
            pandas.DataFrame: Loaded sents
        """
        feat_path = os.path.join(dataset_path, f'{token}.{suf}')
        field = getattr(self, field_name, None)
        sents_field = getattr(self, sents_field_name, None)

        if os.path.isfile(feat_path):
            feat = self._load_feat(feat_path, suf)
            self.logger.debug(f'[{suf}] feature loaded successfully from [{feat_path}].')
        else:
            feat = None
            self.logger.debug(f'[{feat_path}] not found, [{suf}] features are not loaded.')

        def get_sents(single_sents: list):
            array_size = [self.config['max_sent_num'], self.config['max_sent_len']]
            sents = np.zeros(array_size)
            sent_num = 0
            for s in single_sents:
                sents[sent_num] = np.pad(s, (0, array_size[1] - len(s)))  # sents[idx] 第idx个用户的多个句子组成的 tensor 矩阵
                sent_num += 1
            return sents

        self.wd2id = {
            '[WD_PAD]': 0,
            '[WD_MISS]': 1
        }
        self.id2wd = ['[WD_PAD]', '[WD_MISS]']

        def word_map(sent):
            value = []
            for i, wd in enumerate(sent):
                if wd not in self.wd2id.keys():
                    self.wd2id[wd] = i + 2
                    self.id2wd.append(wd)
                value.append(self.wd2id[wd])
            return value

        if feat is not None and field is None:
            raise ValueError(f'{field_name} must be exist if {suf}_feat exist.')
        if feat is not None and field not in feat:
            raise ValueError(f'{field_name} must be loaded if {suf}_feat is loaded.')
        if feat is not None:
            # tokens = [feat[sents_field].agg(np.concatenate)]
            # tokens = np.concatenate(tokens)
            # new_word_ids_list =
            feat[sents_field] = feat[sents_field].apply(word_map)
            feat = feat.groupby(field).apply(lambda x: get_sents([i for i in x[sents_field]])).to_frame()
            feat.reset_index(inplace=True)
            feat.columns = [field, sents_field]

        return feat

    def filter_data_with_no_sents(self):
        """Remove interactions without text from both sides

        """
        self.inter_feat = self.inter_feat[self.inter_feat[self.uid_field].isin(self.user_sents[self.uid_field])]
        self.inter_feat = self.inter_feat[self.inter_feat[self.iid_field].isin(self.item_sents[self.iid_field])]

    def join(self, df):
        """Given interaction feature, join user/item sents into it.

        Args:
            df (Interaction): Interaction feature to be joint.

        Returns:
            Interaction: Interaction feature after joining operation.
        """
        df = super(PJFDataset, self).join(df)
        if self.user_sents is not None and self.uid_field in df:
            df.update(self.user_sents[df[self.uid_field]])
        if self.item_sents is not None and self.iid_field in df:
            df.update(self.item_sents[df[self.iid_field]])
        return df

    def field2feats(self, field):
        feats = super(PJFDataset, self).field2feats(field)
        if field == self.uid_field:
            feats = [self.inter_feat]
            if self.user_sents is not None:
                feats.append(self.user_sents)
        elif field == self.iid_field:
            if self.item_sents is not None:
                feats.append(self.item_sents)
        return feats

    def _sents_dataframe_to_interaction(self, data):
        new_data = {}
        for k in data:
            value = data[k].values
            if k in [self.uid_field, self.iid_field]:
                new_data[k] = torch.LongTensor(value)
            else:
                new_data[k] = value
        return Interaction(new_data)
