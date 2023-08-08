########################### the parameter for ogbl-citation2 under the existing setting:

## CN
python main_heuristic_ogb.py --data_name ogbl-citation2 --use_heuristic CN

## AA
python main_heuristic_ogb.py --data_name ogbl-citation2 --use_heuristic AA

## RA
python main_heuristic_ogb.py --data_name ogbl-citation2 --use_heuristic RA

## shortest path
python main_heuristic_ogb.py --data_name ogbl-citation2 --use_heuristic shortest_path

## katz
python main_heuristic_ogb.py --data_name ogbl-citation2 --use_heuristic katz_apro --path_len 2

##node2vec
python main_gnn_ogb.py --data_name ogbl-citation2 --gnn_model mlp_model --cat_n2v_feat --hidden_channels 128 --lr 0.001 --dropout 0.  --num_layers 3 --num_layers_predictor 3   --epochs 100 --kill_cnt 20  --batch_size 65536 

## MF
python main_MF_citation2.py --data_name ogbl-citation2   --hidden_channels 128 --lr 0.01 --dropout 0.5  --num_layers 3 --epochs 300 --kill_cnt 20  --eval_steps 1 --batch_size 65536 

## mlp
python main_gnn_ogb.py --data_name ogbl-citation2 --gnn_model mlp_model --hidden_channels 128 --lr 0.001 --dropout 0.  --num_layers 3 --num_layers_predictor 3  --epochs 100 --kill_cnt 20  --batch_size 65536 

## gcn
python main_gnn_ogb.py --data_name ogbl-citation2 --gnn_model GCN --hidden_channels 128 --lr 0.001 --dropout 0.3  --num_layers 3 --num_layers_predictor 3 --epochs 50 --kill_cnt 20  --batch_size 65536 

## sage
python main_gnn_ogb.py --data_name ogbl-citation2 --gnn_model SAGE --hidden_channels 128 --lr 0.001 --dropout 0.3  --num_layers 3 --num_layers_predictor 3 --epochs 50 --kill_cnt 20  --batch_size 65536 


## neognn model layer=2 due to OOM
python main_neognn_ogb.py --data_name ogbl-citation2  --test_batch_size 4096 --lr 0.001 --dropout 0.3  --num_layers 2 --num_layers_predictor 2  --hidden_channels 128  --epochs 50 --kill_cnt 20   --eval_steps 1  --batch_size 32768 

## buddy
python main_buddy_ogb.py --data_name ogbl-citation2  --cache_subgraph_features --lr 0.001 --label_dropout 0.5  --feature_dropout 0.5  --hidden_channels 128  --epochs 50  --eval_steps 1 --kill_cnt 20 --batch_size 65536  --num_negs 1 --sign_dropout 0.2 --sign_k 3 --model BUDDY  

## ncn
python main_ncn_citation2.py  --predictor cn1 --dataset citation2  --xdp 0.0 --tdp 0.3 --gnnedp 0.0 --preedp 0.0  --gnnlr 0.001 --prelr 0.001  --predp 0. --gnndp 0.  --mplayers 3 --hiddim 128 --epochs 20 --kill_cnt 20 --batch_size 65536 --ln --lnnn  --model puregcn --res --testbs 65536 --use_xlin --tailact --proboffset 4.7 --probscale 7.0 --pt 0.3 --trndeg 128 --tstdeg 128 

## ncnc
python main_ncn_citation2.py  --predictor incn1cn1 --dataset citation2  --xdp 0.0 --tdp 0.3 --gnnedp 0.0 --preedp 0.0  --gnnlr 0.001 --prelr 0.001  --predp 0.3 --gnndp 0.3  --mplayers 3 --hiddim 128 --epochs 20 --kill_cnt 20 --batch_size 32768 --ln --lnnn  --model puregcn --res --testbs 65536 --use_xlin --tailact --proboffset 4.7 --probscale 7.0 --pt 0.3 --trndeg 128 --tstdeg 128  

##seal

##train  --output_dir specify the path you want to save the model 
python main_seal_ogb.py --data_name ogbl-citation2  --output_dir 'output_test' --save  --batch_size 32 --num_hops 1  --lr 0.001 --hidden_channels 128  --num_layers 3 --use_feature --use_edge_weight --eval_steps 1 --epochs 10 --kill_cnt 20 --dynamic_train --dynamic_val --dynamic_test --train_percent 2 --val_percent 1 --test_percent 1

##test: get the test performance for one seed, you need to specify the --model_path: the path of the saved model, --test_seed: the seed number , if you add want to get the validation results, add --val_full
python main_seal_ogb.py --runs 1 --model_path 'output_test' --test_seed 0 --lr 0.001  --test_multiple_models  --data_name ogbl-citation2 --num_workers 4  --batch_size 32 --num_hops 1   --hidden_channels 128  --num_layers 3 --use_feature --use_edge_weight --eval_steps 1 --epochs 10 --kill_cnt 20 --dynamic_train --dynamic_val --dynamic_test --train_percent 2 --val_percent 100 --test_percent 100 