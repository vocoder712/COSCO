"""
结果持久化工具 / result-persistence utilities.

中文说明:
    - `save_to_dataframe` 把单次运行结果以一行追加到汇总 CSV.
    - `save_to_file_directory` 在 <save_dir>/<dataset>/<shot>-shot/ 下写一份
      可读性更高的 results_sam_proto_neg.txt 文件.

English:
    - `save_to_dataframe` appends one row of summary metrics to the
      aggregated CSV file.
    - `save_to_file_directory` writes a human-readable
      results_sam_proto_neg.txt under <save_dir>/<dataset>/<shot>-shot/.
"""

import os
import pandas as pd
import os
import argparse


def save_to_dataframe(acc, args):
    """
    追加一行到 args.save_dir/args.save_name 指定的汇总 CSV.

    Append one row of (Dataset, Shots, Normalization, mean accuracy)
    to the aggregated CSV located at args.save_dir/args.save_name.
    """
    # 汇总 CSV 路径 / path to the aggregated CSV
    path = args.save_dir + '/' + args.save_name

    # 构造一行 DataFrame / build a single-row DataFrame
    new_data = {
        'Dataset': [args.dataset],
        'Shots': [args.shot],
        'Normalization': [args.normalize],
        'Result': [acc.mean()]
    }

    new_df = pd.DataFrame(new_data)

    # 以追加模式写入, 不写 header / append without header
    new_df.to_csv(path, mode='a', header=False, index=False)


def save_to_file_directory(acc, args):
    """
    在 <save_dir>/<dataset>/<shot>-shot/ 下写 results_sam_proto_neg.txt.

    Write a results_sam_proto_neg.txt file under
    <save_dir>/<dataset>/<shot>-shot/ that contains the per-run accuracies
    plus their mean.
    """
    data_dir = args.save_dir
    dataset_name = args.dataset
    shot_dir = args.shot
    normalize_data = args.normalize

    # 目标路径并按需创建 / target dir, created if missing
    path = os.path.join(data_dir, dataset_name, str(shot_dir) + '-shot')
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, 'results_sam_proto_neg.txt'), 'w') as f:
        f.write(dataset_name + '\n')
        f.write(str(shot_dir) + '\n')
        f.write(str(normalize_data) + '\n')
        f.write(str(acc) + '\n')
        f.write(str(acc.mean()))
