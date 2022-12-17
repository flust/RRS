# @Time   : 2022/3/2
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn


import argparse
import os

import logging
from logging import getLogger

from recbole.utils import init_logger, init_seed, set_color

from recbole_pjf.config import PJFConfig
from recbole_pjf.data import create_dataset, data_preparation
from recbole_pjf.utils import get_model, get_trainer


def run_recbole_pjf(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True, saved_model_file=None):
    # configurations initialization
    config = PJFConfig(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'],
                          multi_direction=config['biliteral'])(config, model)

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, model_file=saved_model_file, show_progress=config['show_progress'])

#     logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        # 'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        # 'best_valid_result': best_valid_result,
        'test_result': test_result
    }



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BPR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='tech', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--saved_model_file', type=str, default='/code/yangchen/RecBole-PJF/saved/BPR-Nov-30-2022_20-28-28.pth', help='saved model file')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_pjf(model=args.model, dataset=args.dataset, config_file_list=config_file_list, saved_model_file=args.saved_model_file)

    

    