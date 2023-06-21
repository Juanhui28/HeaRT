########################### the parameter for ogbl-ppa under the HeaRT:

## CN
python main_heuristic_ogb.py --data_name ogbl-ppa --use_heuristic CN

## AA
python main_heuristic_ogb.py --data_name ogbl-ppa --use_heuristic AA

## RA
python main_heuristic_ogb.py --data_name ogbl-ppa --use_heuristic RA

## shortest path
python main_heuristic_ogb.py --data_name ogbl-ppa --use_heuristic shortest_path

## katz
python main_heuristic_ogb.py --data_name ogbl-ppa --use_heuristic katz_close 

##node2vec
python main_gnn_ogb.py --data_name ogbl-ppa --gnn_model mlp_model --cat_n2v_feat --lr 0.001  --dropout 0  --num_layers 3 --num_layers_predictor 3   --hidden_channels 256  --epochs 9999 --eval_steps 1 --kill_cnt 100  --batch_size 65536 

## MF
python main_MF_collabppa.py --data_name ogbl-ppa  --lr 0.01 --dropout 0.3 --num_layers 3 --hidden_channels 256  --epochs 9999   --eval_steps 1 --kill_cnt 100  --batch_size 65536 

## mlp
python main_gnn_ogb.py --data_name ogbl-ppa --gnn_model mlp_model --lr 0.001 --dropout 0.5 --num_layers 3 --num_layers_predictor 3   --hidden_channels 256  --epochs 9999 --eval_steps 1 --kill_cnt 100  --batch_size 65536 

## gcn
python main_gnn_ogb.py  --data_name ogbl-ppa  --gnn_model GCN --lr 0.001 --dropout 0.3 --num_layers 3 --hidden_channels 256  --num_layers_predictor 3 --epochs 9999 --kill_cnt 100 --eval_steps 1  --batch_size 65536  

## sage
python main_gnn_ogb.py  --data_name ogbl-ppa  --gnn_model SAGE --lr 0.001 --dropout 0 --num_layers 3 --hidden_channels 256  --num_layers_predictor 3 --epochs 9999 --kill_cnt 100 --eval_steps 1  --batch_size 65536  

## neognn
python main_neognn_ogb.py  --data_name ogbl-ppa --lr 0.01 --dropout 0.3 --l2 0 --num_layers 3 --num_layers_predictor 3 --hidden_channels 256 --epochs 9999  --kill_cnt 20  --batch_size 1024  

## buddy
python main_buddy_ogb.py  --cache_subgraph_features --data_name ogbl-ppa  --lr 0.001  --l2 0 --label_dropout 0.5   --feature_dropout 0.5  --hidden_channels 256  --epochs 9999  --eval_steps 1 --kill_cnt 20   --batch_size 65536  --use_RA true  --add_normed_features 1 --use_feature 0     --model BUDDY 

## ncn
python main_ncn_ogb.py --predictor cn1 --dataset ppa  --xdp 0.0 --tdp 0.0 --gnnedp 0.1 --preedp 0.0  --gnnlr 0.001 --prelr 0.001 --predp 0 --gnndp 0 --mplayers 3 --hiddim 256 --epochs 9999 --kill_cnt 20 --batch_size 65536  --ln --lnnn   --model gcn --maskinput  --tailact  --res  --testbs 2048 --proboffset 8.5 --probscale 4.0 --pt 0.1 --alpha 0.9 --splitsize 131072 

## ncnc
python main_ncn_ogb.py --predictor incn1cn1  --dataset ppa --xdp 0.0 --tdp 0.0 --gnnedp 0.1 --preedp 0.0  --gnnlr 0.001 --prelr 0.001 --predp 0 --gnndp 0 --mplayers 3 --hiddim 256 --epochs 15 --kill_cnt 20 --batch_size 65536  --ln --lnnn  --model gcn --maskinput  --tailact  --res  --testbs 2048 --proboffset 8.5 --probscale 4.0 --pt 0.1 --alpha 0.9 --splitsize 131072  

##seal
#train --output_dir specify the path you want to save the model 
python  main_seal_ogb.py --save   --output_dir 'output_test'   --data_name ogbl-ppa  --lr 0.001  --num_layers 3  --hidden_channels 256 --dynamic_train --dynamic_val --dynamic_test  --num_hops 1 --use_feature --use_edge_weight  --train_percent 5  --val_percent 1 --test_percent 1  --epochs 15 --kill_cnt 20  --batch_size 32 --test_bs 8 

#test: get the test performance for one seed, you need to specify the --model_path: the path of the saved model, --test_seed: the seed number , if you add want to get the validation results, add --val_full
python  main_seal_ogb.py  --data_name ogbl-ppa  --runs 1 --model_path 'output_test' --test_seed 0  --test_multiple_models --lr 0.001 --test_seed 0  --runs 1 --num_layers 3  --hidden_channels 256 --dynamic_train --dynamic_val --dynamic_test  --num_hops 1 --use_feature --use_edge_weight  --train_percent 5  --val_percent 100 --test_percent 100  --epochs 15 --kill_cnt 20  --batch_size 32 --test_bs 8  