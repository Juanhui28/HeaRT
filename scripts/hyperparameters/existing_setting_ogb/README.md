# Hyperparameters for ogbl-collab, ogbl-ddi, ogbl-ppa and ogbl-citation2 under the existing setting

`ogbl-collab.sh`, `ogbl-ddi.sh`, `ogbl-ppa.sh`, `ogbl-citation2.sh` provide the parameters for  ogbl-collab, ogbl-ddi, ogbl-ppa and ogbl-citation2 under the existing setting. 


For some large models such as SEAL and NCNC, it's recommended to run 10 seeds in parallel.  To run a single seed, for example, you can add `--runs 1  --seed 0`. One example to run a single seed for SEAL on ogbl-collab is:
```
python main_seal_ogb.py --use_valedges_as_input  --data_name ogbl-collab  --lr  0.001  --num_layers 3  --num_layers_predictor 3 --hidden_channels 256 --epochs 9999  --eval_steps 1  --kill_cnt 20  --batch_size 32   --num_hops 1 --train_percent 15 --runs 1  --seed 0 
```

Some noteworthy cases to consider are:
- ogbl-ppa

 NCNC, SEAL: Each epoch and evaluation process can take several hours to complete. Therefore, it would be quite time-consuming to stop training if the validation performance does not improve within 20 checkpoints. For NCNC, We have defined a maximum number of epochs for training. For SEAL, we follow the setting in the ["original implementation"](https://github.com/facebookresearch/SEAL_OGB/tree/main) to set a maximum epoch number and evaluate every 5 epoch.

-  ogbl-citation2

SEAL: Due to the significant time complexity,
We follow the setting in the ["original implementation"](https://github.com/facebookresearch/SEAL_OGB/tree/main) to train and evaluate only a portion of training edges and evaluation edges.  It has two steps: 1) evaluate 1% validation data in parallel and save the best epoch's model for each seed; 2) evaluate each best model on all test edges

The first step: please first create a directory `output_test` for the following example:
```
python main_seal_ogb.py --data_name ogbl-citation2  --output_dir 'output_test' --save  --batch_size 32 --num_hops 1  --lr 0.001 --hidden_channels 128  --num_layers 3 --use_feature --use_edge_weight --eval_steps 1 --epochs 10 --kill_cnt 20 --dynamic_train --dynamic_val --dynamic_test --train_percent 2 --val_percent 1 --test_percent 1
```

The second step: you need to specify the `--model_path`: the path of the saved model, `--test_seed`: the seed number . The default setting is only getting the test result,  if you add want to get the validation results, add  `--val_full`

```
python main_seal_ogb.py --runs 1 --model_path 'output_test' --test_seed 0 --lr 0.001  --test_multiple_models  --data_name ogbl-citation2 --num_workers 4  --batch_size 32 --num_hops 1   --hidden_channels 128  --num_layers 3 --use_feature --use_edge_weight --eval_steps 1 --epochs 10 --kill_cnt 20 --dynamic_train --dynamic_val --dynamic_test --train_percent 2 --val_percent 100 --test_percent 100 
```




