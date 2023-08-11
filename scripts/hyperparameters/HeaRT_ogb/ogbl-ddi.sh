

########################### the parameter for ogbl-ddi under HeaRT:

## CN
python main_heuristic_ogb.py --data_name ogbl-ddi --use_heuristic CN

## AA
python main_heuristic_ogb.py --data_name ogbl-ddi --use_heuristic AA

## RA
python main_heuristic_ogb.py --data_name ogbl-ddi --use_heuristic RA

## shortest path
python main_heuristic_ogb.py --data_name ogbl-ddi --use_heuristic shortest_path

## katz
python main_heuristic_ogb.py --data_name ogbl-ddi --use_heuristic katz_apro --path_len 2

## node2vec
python main_node2vec_ddi.py   --data_name ogbl-ddi   --lr 0.01 --dropout 0 --num_layers 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536  

## MF
python main_MF_ddi.py  --data_name ogbl-ddi   --lr 0.01 --dropout 0.5  --num_layers 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536  

## gcn
python main_gnn_ddi.py  --data_name ogbl-ddi   --gnn_model GCN --lr 0.01 --dropout 0 --num_layers 3 --hidden_channels 256  --num_layers_predictor 3 --epochs 9999 --kill_cnt 100 --eval_steps 1  --batch_size 65536    

## GAT
python main_gnn_ddi.py  --data_name ogbl-ddi   --gnn_model GAT --lr 0.001 --dropout 0.3 --num_layers 3 --hidden_channels 256  --num_layers_predictor 3 --epochs 9999 --kill_cnt 100 --eval_steps 1  --batch_size 65536    

## SAGE
python main_gnn_ddi.py  --data_name ogbl-ddi   --gnn_model SAGE --lr 0.001 --dropout 0.3 --num_layers 3 --hidden_channels 256  --num_layers_predictor 3 --epochs 9999 --kill_cnt 100 --eval_steps 1  --batch_size 65536    

## gae
python main_gae_ddi.py  --data_name ogbl-ddi   --lr 0.001 --dropout 0.3 --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536   

## neognn
python main_neognn_ddi.py --data_name ogbl-ddi   --lr 0.01 --dropout 0  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 20 --batch_size 4096 --test_batch_size 4096  

## buddy
python main_buddy_ddi.py --data_name ogbl-ddi  --cache_subgraph_features  --train_node_embedding --propagate_embeddings --hidden_channels 256 --lr 0.01  --l2 0 --label_dropout 0.5  --feature_dropout 0.5 --num_negs 1 --use_feature 0 --sign_k 2  --model BUDDY --epochs 9999 --eval_steps 1 --kill_cnt 20  --batch_size 65536 

## peg
python main_peg_ddi.py --PE_method 'DW'  --data_name ogbl-ddi --lr 0.01  --dropout 0.3 --num_layers 3  --hidden_channels 256 --epochs 9999  --eval_steps 1 --kill_cnt 20   --batch_size 65536

## ncn
python main_ncn_ddi.py  --predictor cn1  --dataset ddi --xdp 0.05 --tdp 0.0 --gnnedp 0.0 --preedp 0.0  --gnnlr 0.01 --prelr 0.01 --predp 0.3 --gnndp 0.3 --mplayers 3 --hiddim 256 --epochs 9999 --kill_cnt 20  --batch_size 65536  --ln --lnnn --model puresum   --testbs 1024  --splitsize 1024 --use_xlin  --twolayerlin  --res  --maskinput 

## seal
#train --output_dir specify the path you want to save the model 
python main_seal_ddi.py  --data_name ogbl-ddi --output_dir 'output_test' --save   --lr 0.001  --num_layers 3  --hidden_channels 256 --dynamic_train --dynamic_val --dynamic_test  --num_hops 1  --use_edge_weight --ratio_per_hop 0.2  --train_percent 1  --val_percent 1 --test_percent 1  --epochs 50 --kill_cnt 20  --batch_size 32 --test_bs 16

#test: get the test performance for one seed, you need to specify the --model_path: the path of the saved model, --test_seed: the seed number , if you add want to get the validation results, add --val_full
python main_seal_ddi.py  --data_name ogbl-ddi  --runs 1 --model_path 'output_test' --test_seed 0 --test_multiple_models --lr 0.001 --num_layers 3  --hidden_channels 256 --dynamic_train --dynamic_val --dynamic_test  --num_hops 1  --use_edge_weight --ratio_per_hop 0.2  --train_percent 1  --val_percent 100 --test_percent 100 --epochs 10 --kill_cnt 20  --batch_size 32 --test_bs 32 

        