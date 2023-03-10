# @Time   : 2022/4/18
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
recbole_pjf
"""
import os
os.system("pip install hyperopt==0.2.5 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com")
import argparse

from recbole.trainer import HyperTuning
from recbole_pjf.quick_start import objective_function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', type=str, default=None, help='fixed config files')
    parser.add_argument('--params_file', type=str, default=None, help='parameters file')
    parser.add_argument('--output_file', type=str, default='hyper_example.result', help='output file')
    args, _ = parser.parse_known_args()

    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    hp = HyperTuning(objective_function, algo='exhaustive', early_stop=10, max_evals=100,
                     params_file=args.params_file, fixed_config_file_list=config_file_list)
    hp.run()
    with open(args.output_file, "w") as fp:
        for params in hp.params2result:
            fp.write(params + "\n")
            fp.write(
                "Valid result:\n"
                + str(hp.params2result[params]["best_valid_result"])
                + "\n"
            )

            fp.write(
                "Test result:\n"
                + str(hp.params2result[params]["test_result"])
                + "\n\n"
            )

    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])


if __name__ == '__main__':
    main()
