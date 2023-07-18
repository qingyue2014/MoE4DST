python T5_cluster.py --train_batch_size 32 --GPU 2 --data_dir data --model_name t5-base --dataset mwz --gradient_accumulation_steps 4 --max_length 512 --n_epochs 10 
--except_domain attraction --slot_lang human --mode train --max_history 20 --clu_algorithm kmeans --clu_encoder t5-base --cluster_dir ClusterData --saving save/ --freeze_transformer --adapter
