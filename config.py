# Copyright (c) Facebook, Inc. and its affiliates

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saving_dir", type=str, default="save", help="Path for saving")
    parser.add_argument("--cluster_dir", type=str, default="ClusterData", help="Path for cluster data")
    parser.add_argument("--dataset", type=str, default="mwz", help="sgd or mwz")
    parser.add_argument("--data_dir", type=str, default="data", help="aug_data or data")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--dev_batch_size", type=int, default=16, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Batch size for test")
    parser.add_argument("--num_workers", type=int, default=0, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action='store_true', help="continual baseline")
    parser.add_argument("--length", type=int, default=50, help="Batch size for validation")
    parser.add_argument("--max_history", type=int, default=1, help="max number of turns in the dialogue")
    parser.add_argument("--max_length", type=int, default=512, help="max length of the dialogue")
    parser.add_argument("--GPU", type=int, default=8, help="number of gpu to use")
    parser.add_argument("--model_name", type=str, default="t5-base", help="use t5 or bart?")
    parser.add_argument("--slot_lang", type=str, default="human", help="use 'none', 'human', 'naive', 'value', 'question', 'slottype' slot description")
    parser.add_argument("--fewshot", type=float, default=0.0, help="data ratio for few shot experiment")
    parser.add_argument("--fix_label", action='store_true')
    parser.add_argument("--except_domain", type=str, default="taxi", help="hotel, train, restaurant, attraction, taxi")
    parser.add_argument("--only_domain", type=str, default="none", help="hotel, train, restaurant, attraction, taxi")
    parser.add_argument("--semi", action='store_true')
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--do_test", action="store_true", help="do test.")
    parser.add_argument("--use_shuffle", action="store_true", help="shuffle the training data")
    parser.add_argument("--clu_encoder", type=str, default="t5-base", help="t5-base, t5-small, bert, bart-base, roberta-base")
    parser.add_argument("--clu_algorithm", type=str, default="kmeans", help="gmm, agg, brich, kmeans")
    #adapter
    parser.add_argument("--freeze_transformer", action="store_true", help="Freeze Transformer parameters.")
    parser.add_argument(
        "--adapter", action="store_true", help="Use a single adapter."
    )
    parser.add_argument("--adapter_name", type=str, default="base_adapter", help="A name for the adapter.")
    parser.add_argument("--adapter_config", type=str, default="houlsby", help="")
    parser.add_argument("--freeze_adapter", action="store_true", help="Freeze adapter parameters.")
    #ensemble
    parser.add_argument("--dist_way", type=str, default="euc", help="[inner, euc, average]")
    parser.add_argument("--dist_format", type=str, default="soft", help="one-hot or soft")
    parser.add_argument("--T", type=float, default=2)
    parser.add_argument("--class_id", type=int, default=-1, help="class id")
    parser.add_argument("--sub_id", type=int, default=-1, help="sub id")
    args = parser.parse_args()
    return args
