
########################### the parameter for ogbl-collab under the existing setting:

## CN
python main_heuristic_ogb.py --data_name ogbl-collab --use_heuristic CN --use_valedges_as_input

## AA
python main_heuristic_ogb.py --data_name ogbl-collab --use_heuristic AA --use_valedges_as_input

## RA
python main_heuristic_ogb.py --data_name ogbl-collab --use_heuristic RA --use_valedges_as_input

## shortest path
python main_heuristic_ogb.py --data_name ogbl-collab --use_heuristic shortest_path --use_valedges_as_input

## katz
python main_heuristic_ogb.py --data_name ogbl-collab --use_heuristic katz_close --use_valedges_as_input



##node2vec
python main_gnn_ogb.py  --cat_n2v_feat --data_name ogbl-collab --gnn_model mlp_model --hidden_channels 256 --lr 0.001 --dropout 0.  --num_layers 3 --num_layers_predictor 3  --epochs 9999 --kill_cnt 100  --batch_size 65536 

## MF
python main_MF_collabppa.py --data_name ogbl-collab  --hidden_channels 256 --lr 0.01 --dropout 0.3  --num_layers 3  --epochs 9999 --kill_cnt 100 --eval_steps 1  --batch_size 65536

## mlp
python main_gnn_ogb.py   --data_name ogbl-collab  --gnn_model mlp_model --hidden_channels 256 --lr 0.001 --dropout 0 --num_layers 3 --num_layers_predictor 3 --epochs 9999 --kill_cnt 100  --batch_size 65536 


## gcn
python main_gnn_ogb.py  --use_valedges_as_input  --data_name ogbl-collab  --gnn_model GCN --hidden_channels 256 --lr 0.001 --dropout 0.  --num_layers 3 --num_layers_predictor 3 --epochs 9999 --kill_cnt 100  --batch_size 65536 

## gat
python main_gnn_ogb.py  --use_valedges_as_input  --data_name ogbl-collab --gnn_model GAT --hidden_channels 256 --lr 0.001 --dropout 0.  --num_layers 3 --num_layers_predictor 3  --epochs 9999 --kill_cnt 100  --batch_size 65536 

## sage
python main_gnn_ogb.py  --use_valedges_as_input  --data_name ogbl-collab --gnn_model SAGE --hidden_channels 256 --lr 0.001 --dropout 0.3  --num_layers 3 --num_layers_predictor 3  --epochs 9999 --kill_cnt 100  --batch_size 65536 


##neognn
python main_neognn_ogb.py --use_valedges_as_input   --data_name ogbl-collab  --lr 0.001 --dropout 0.3  --num_layers 3 --num_layers_predictor 3 --hidden_channels 256 --epochs 9999 --kill_cnt 20  --batch_size 1024 

## buddy
python main_buddy_ogb.py --data_name ogbl-collab   --lr 0.01   --label_dropout 0.3  --feature_dropout 0.3 --hidden_channels 256  --epochs 9999 --eval_steps 1 --kill_cnt 20  --batch_size 65536 

## peg
python main_peg_ogb.py --PE_method 'DW' --use_valedges_as_input --data_name ogbl-collab  --lr 0.001  --dropout 0.3 --num_layers 3  --hidden_channels 256 --epochs 9999  --eval_steps 1 --kill_cnt 20   --batch_size 65536

## ncn
python main_ncn_collabppa.py --predictor cn1  --dataset collab  --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0  --gnnlr 0.001  --prelr 0.001 --predp 0.3 --gnndp  0.3  --probscale 2.5 --proboffset 6.0 --alpha 1.05   --mplayers 3  --hiddim 256  --epochs 9999 --kill_cnt 20  --batch_size 65536  --ln --lnnn --model gcn  --testbs 131072  --maskinput --use_valedges_as_input   --res  --use_xlin  --tailact  


## ncnc
python main_ncn_collabppa.py --predictor incn1cn1 --dataset collab  --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0  --gnnlr 0.001  --prelr 0.001 --predp 0.3 --gnndp  0.3  --probscale 2.5 --proboffset 6.0 --alpha 1.05   --mplayers 3  --hiddim 256  --epochs 9999 --kill_cnt 20  --batch_size 65536  --ln --lnnn  --model gcn  --testbs 131072  --maskinput --use_valedges_as_input   --res  --use_xlin  --tailact  

## seal
python main_seal_ogb.py --use_valedges_as_input  --data_name ogbl-collab  --lr  0.001  --num_layers 3  --num_layers_predictor 3 --hidden_channels 256 --epochs 9999  --eval_steps 1  --kill_cnt 20  --batch_size 32   --num_hops 1 --train_percent 15 

