# Hyperparameters for ogbl-collab, ogbl-ddi, ogbl-ppa and ogbl-citation2 under HeaRT

`ogbl-collab.sh`, `ogbl-ddi.sh`, `ogbl-ppa.sh`, `ogbl-citation2.sh` provide the parameters for  ogbl-collab, ogbl-ddi, ogbl-ppa and ogbl-citation2 under HeaRT.

Since we have more negative samples in OGB datasets, we follow the setting in the ["original implementation"](https://github.com/facebookresearch/SEAL_OGB/tree/main) to train and evaluate only a portion of training edges and evaluation edges.  It has two steps: 1) evaluate 1% validation data in parallel and save the best epoch's model for each seed; 2) evaluate each best model on all test edges. One example on ogbl-collab is:

The first step: please first create a directory `output_test` for the following example:
```
python main_seal_ogb.py  --save   --output_dir 'output_test'   --lr 0.001 --num_layers 3  --hidden_channels 256  --data_name ogbl-collab  --num_hops 1 --train_percent 15 --dynamic_train --dynamic_val --dynamic_test --val_percent 1 --test_percent 1  --use_valedges_as_input  --epochs 9999 --kill_cnt 20  --batch_size 32  --test_bs 64 
```


The second step: you need to specify the `--model_path`: the path of the saved model, `--test_seed`: the seed number . The default setting is only getting the test result,  if you add want to get the validation results, add  `--val_full`

```
python main_seal_ogb.py  --data_name ogbl-collab  --runs 1 --model_path 'output_test' --test_seed 0  --test_multiple_models --lr 0.001 --num_layers 3  --hidden_channels 256    --num_hops 1 --train_percent 15 --dynamic_train --dynamic_val --dynamic_test --val_percent 100 --test_percent 100  --use_valedges_as_input  --epochs 9999 --kill_cnt 20  --batch_size 32  --test_bs 32 
```


