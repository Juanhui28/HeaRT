########################### the parameter for ogbl-collab under HeaRT:

## CN
python main_heuristic_ogb.py --data_name ogbl-collab --use_heuristic CN --use_valedges_as_input

## AA
python main_heuristic_ogb.py --data_name ogbl-collab --use_heuristic AA --use_valedges_as_input

## RA
python main_heuristic_ogb.py --data_name ogbl-collab --use_heuristic RA --use_valedges_as_input

## shortest path
python main_heuristic_ogb.py --data_name ogbl-collab --use_heuristic shortest_path --use_valedges_as_input

## katz
python main_heuristic_ogb.py --data_name ogbl-collab --use_heuristic katz_apro --use_valedges_as_input --path_len 2

## node2vec
python main_gnn_ogb.py  --data_name ogbl-collab --cat_n2v_feat --gnn_model mlp_model  --use_valedges_as_input   --lr 0.001 --dropout 0 --num_layers 3 --hidden_channels 256  --num_layers_predictor 3 --epochs 9999 --kill_cnt 100 --eval_steps 1  --batch_size 65536  

## MF
python main_MF_collabppa.py  --data_name ogbl-collab --lr 0.001 --dropout 0. --num_layers 3 --hidden_channels 256  --epochs 9999 --kill_cnt 100 --eval_steps 1  --batch_size 65536  

## mlp
python main_gnn_ogb.py  --data_name ogbl-collab  --use_valedges_as_input --gnn_model mlp_model  --lr 0.001 --dropout 0 --num_layers 3 --hidden_channels 256  --num_layers_predictor 3 --epochs 9999 --kill_cnt 100 --eval_steps 1  --batch_size 65536  

## gcn
python main_gnn_ogb.py  --data_name ogbl-collab  --use_valedges_as_input --gnn_model GCN  --lr 0.001 --dropout 0.3 --num_layers 3 --hidden_channels 256  --num_layers_predictor 3 --epochs 9999 --kill_cnt 100 --eval_steps 1  --batch_size 65536  

## GAT
python main_gnn_ogb.py  --data_name ogbl-collab  --use_valedges_as_input --gnn_model GAT  --lr 0.001 --dropout 0 --num_layers 3 --hidden_channels 256  --num_layers_predictor 3 --epochs 9999 --kill_cnt 100 --eval_steps 1  --batch_size 65536  

## SAGE
python main_gnn_ogb.py  --data_name ogbl-collab  --use_valedges_as_input --gnn_model SAGE  --lr 0.001 --dropout 0. --num_layers 3 --hidden_channels 256  --num_layers_predictor 3 --epochs 9999 --kill_cnt 100 --eval_steps 1  --batch_size 65536  

## neognn
python main_neognn_ogb.py  --data_name ogbl-collab  --use_valedges_as_input --use_2hop  --lr 0.01 --dropout 0.5 --num_layers 3 --hidden_channels 256  --num_layers_predictor 3 --epochs 9999 --kill_cnt 20 --eval_steps 1  --batch_size 1024 --test_batch_size 4096   

## buddy
python main_buddy_ogb.py  --data_name ogbl-collab  --lr 0.001 --l2 0  --label_dropout 0  --feature_dropout 0 --hidden_channels 256  --epochs 9999 --eval_steps 1 --kill_cnt 20 --save  --batch_size 65536 --model BUDDY

## peg
python main_peg_ogb.py --PE_method 'DW' --use_valedges_as_input  --data_name ogbl-collab --lr 0.01  --dropout 0 --num_layers 3  --hidden_channels 256 --epochs 9999  --eval_steps 1 --kill_cnt 20   --batch_size 65536

## ncn
python main_ncn_ogb.py --dataset collab --predictor cn1   --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --gnnlr  0.001  --prelr 0.001  --predp 0.3 --gnndp  0.3   --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 9999 --kill_cnt 20  --batch_size 65536  --ln --lnnn  --model gcn  --testbs 2048 --splitsize 16384 --maskinput --use_valedges_as_input   --res  --use_xlin  --tailact  

## ncnc
python main_ncn_ogb.py --dataset collab --predictor incn1cn1   --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --gnnlr  0.001  --prelr 0.001  --predp 0 --gnndp  0   --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 9999 --kill_cnt 20  --batch_size 65536  --ln --lnnn  --model gcn  --testbs 2048 --splitsize 16384 --maskinput --use_valedges_as_input   --res  --use_xlin  --tailact  

## seal
#train --output_dir specify the path you want to save the model 
python main_seal_ogb.py  --save   --output_dir 'output_test'   --lr 0.001 --num_layers 3  --hidden_channels 256  --data_name ogbl-collab  --num_hops 1 --train_percent 15 --dynamic_train --dynamic_val --dynamic_test --val_percent 1 --test_percent 1  --use_valedges_as_input  --epochs 9999 --kill_cnt 20  --batch_size 32  --test_bs 64 

#test: get the test performance for one seed, you need to specify the --model_path: the path of the saved model, --test_seed: the seed number , if you add want to get the validation results, add --val_full
python main_seal_ogb.py  --data_name ogbl-collab  --runs 1 --model_path 'output_test' --test_seed 0  --test_multiple_models --lr 0.001 --num_layers 3  --hidden_channels 256    --num_hops 1 --train_percent 15 --dynamic_train --dynamic_val --dynamic_test --val_percent 100 --test_percent 100  --use_valedges_as_input  --epochs 9999 --kill_cnt 20  --batch_size 32  --test_bs 32 