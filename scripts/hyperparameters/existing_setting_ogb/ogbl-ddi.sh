

########################### the parameter for ogbl-ddi under the existing setting:

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

##node2vec
python main_node2vec_ddi.py --data_name ogbl-ddi --lr 0.001 --dropout 0.3  --num_layers 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536  

## MF
python  main_MF_ddi.py  --data_name ogbl-ddi   --lr 0.01 --dropout 0.3 --num_layers 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536  

## gcn
python main_gnn_ddi.py --data_name ogbl-ddi --gnn_model GCN  --lr 0.01 --dropout 0.5  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536 

## gat
python main_gnn_ddi.py --data_name ogbl-ddi --gnn_model GAT  --lr 0.001 --dropout 0.5  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536   

## sage
python main_gnn_ddi.py --data_name ogbl-ddi --gnn_model SAGE  --lr 0.001 --dropout 0.3  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536   

## gae
python main_gae_ddi.py --data_name ogbl-ddi --lr 0.001 --dropout 0.  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536  

## neognn
python main_neognn_ddi.py --data_name ogbl-ddi --lr 0.01 --dropout 0  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 20 --batch_size 8192  

## buddy
python main_buddy_ddi.py --data_name ogbl-ddi  --lr 0.001 --label_dropout 0.5  --feature_dropout 0.5  --hidden_channels 256  --epochs 9999 --eval_steps 1 --kill_cnt 20  --batch_size 65536  --train_node_embedding --propagate_embeddings --num_negs 1 --use_feature 0 --sign_k 2 --model BUDDY  

## peg
python main_peg_ddi.py --PE_method 'DW' --data_name ogbl-ddi --lr 0.001   --dropout 0.5 --num_layers 3  --hidden_channels 256  --epochs 9999 --eval_steps 1 --kill_cnt 20 --batch_size 65536   

## ncn
python main_ncn_ddi.py --predictor cn1 --dataset ddi  --xdp 0.05 --tdp 0.0 --gnnedp 0.0 --preedp 0.0  --gnnlr 0.01 --prelr 0.01 --predp 0.3 --gnndp 0.3 --mplayers 1 --hiddim 256 --epochs 9999 --kill_cnt 20  --batch_size 65536  --ln --lnnn   --model puresum   --testbs 131072   --use_xlin  --twolayerlin  --res  --maskinput 

## ncnc
python main_ncn_ddi.py  --predictor incn1cn1 --dataset ddi  --xdp 0.05 --tdp 0.0 --gnnedp 0.0 --preedp 0.0  --gnnlr 0.01 --prelr 0.01 --predp 0.3 --gnndp 0.3 --mplayers 1 --hiddim 256 --epochs 9999 --kill_cnt 20  --batch_size 66356   --ln --lnnn  --model puresum  --proboffset 3 --probscale 10 --pt 0.1 --alpha 0.5 --testbs 24576 --splitsize 262144  --use_xlin  --twolayerlin  --res  --maskinput 

## seal
python main_seal_ddi.py  --data_name ogbl-ddi  --lr 0.001  --num_layers 3 --hidden_channels 256 --epochs 9999  --eval_steps 1 --kill_cnt 20  --batch_size 32 --num_hops 1 --ratio_per_hop 0.2 --use_edge_weight --eval_steps 1 --dynamic_val --dynamic_test --train_percent 1 

