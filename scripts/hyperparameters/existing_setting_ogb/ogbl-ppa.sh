
########################### the parameter for ogbl-ppa under the existing setting:

## CN
python main_heuristic_ogb.py --data_name ogbl-ppa --use_heuristic CN

## AA
python main_heuristic_ogb.py --data_name ogbl-ppa --use_heuristic AA

## RA
python main_heuristic_ogb.py --data_name ogbl-ppa --use_heuristic RA

## shortest path
python main_heuristic_ogb.py --data_name ogbl-ppa --use_heuristic shortest_path

## katz
python main_heuristic_ogb.py --data_name ogbl-ppa --use_heuristic katz_apro --path_len 2


##node2vec
python main_gnn_ogb.py --data_name ogbl-ppa  --gnn_model mlp_model --cat_n2v_feat --hidden_channels 256 --lr 0.001 --dropout 0.  --num_layers 3 --num_layers_predictor 3  --epochs 9999 --kill_cnt 100  --batch_size 65536 

## MF
python main_MF_collabppa.py --data_name ogbl-ppa  --lr 0.01 --dropout 0.5 --num_layers 3 --hidden_channels 256  --epochs 9999   --eval_steps 1 --kill_cnt 100  --batch_size 65536 

## mlp
python main_gnn_ogb.py --data_name ogbl-ppa  --gnn_model mlp_model  --hidden_channels 256 --lr 0.001 --dropout 0.  --num_layers 3 --num_layers_predictor 3  --epochs 9999 --kill_cnt 100  --batch_size 65536 


##gcn
python main_gnn_ogb.py --data_name ogbl-ppa --gnn_model GCN --lr 0.01 --dropout 0.3 --num_layers 3 --num_layers_predictor 3  --hidden_channels 256  --epochs 9999 --eval_steps 1 --kill_cnt 100  --batch_size 65536 

## sage
python main_gnn_ogb.py --data_name ogbl-ppa --gnn_model SAGE --lr 0.001 --dropout 0.3 --num_layers 3 --num_layers_predictor 3  --hidden_channels 256  --epochs 9999 --eval_steps 1 --kill_cnt 100  --batch_size 65536 



## neognn
python main_neognn_ogb.py  --data_name ogbl-ppa  --lr 0.01 --dropout 0.  --num_layers 3 --num_layers_predictor 3 --hidden_channels 256 --epochs 9999  --kill_cnt 20  --batch_size 1024   

## buddy
python main_buddy_ogb.py  --cache_subgraph_features --data_name ogbl-ppa  --lr 0.001  --label_dropout 0.  --feature_dropout 0. --hidden_channels 256 --epochs 9999  --eval_steps 1 --kill_cnt 20   --batch_size 65536  --use_RA true  --add_normed_features 1 --use_feature 0  

## ncn
python main_ncn_collabppa.py --predictor cn1  --dataset ppa  --xdp 0.0 --tdp 0.0 --gnnedp 0.1 --preedp 0.0  --gnnlr 0.001 --prelr 0.001 --predp 0 --gnndp 0 --mplayers 3 --hiddim 256 --epochs 9999 --kill_cnt 20 --batch_size 65536  --ln --lnnn  --model gcn --maskinput  --tailact  --res  --testbs 65536 --proboffset 8.5 --probscale 4.0 --pt 0.1 --alpha 0.9 --splitsize 131072 

## ncnc  epoch=50
python main_ncn_collabppa.py --predictor incn1cn1 --dataset ppa  --xdp 0.0 --tdp 0.0 --gnnedp 0.1 --preedp 0.0  --gnnlr 0.001 --prelr 0.001 --predp 0 --gnndp 0 --mplayers 3 --hiddim 256 --epochs 50 --kill_cnt 20 --batch_size 65536  --ln --lnnn --model gcn --maskinput  --tailact  --res  --testbs 65536 --proboffset 8.5 --probscale 4.0 --pt 0.1 --alpha 0.9 --splitsize 131072 

## seal epochs=20
python main_seal_ogb.py  --data_name ogbl-ppa  --lr 0.001  --num_layers 3   --hidden_channels 256  --epochs 20  --eval_steps 5  --kill_cnt 20  --batch_size 32 --runs 1 --num_hops 1 --train_percent 5  --use_feature  --use_edge_weight --num_workers 16  --dynamic_train --dynamic_val --dynamic_test  