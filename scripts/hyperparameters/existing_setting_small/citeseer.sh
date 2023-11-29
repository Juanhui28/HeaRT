
########################### the parameter for citeseer under the existing setting:

## CN
python main_heuristic_CoraCiteseerPubmed.py --data_name citeseer --use_heuristic CN

## AA
python main_heuristic_CoraCiteseerPubmed.py --data_name citeseer --use_heuristic AA

## RA
python main_heuristic_CoraCiteseerPubmed.py --data_name citeseer --use_heuristic RA

## shortest path
python main_heuristic_CoraCiteseerPubmed.py --data_name citeseer --use_heuristic shortest_path

## katz
python main_heuristic_CoraCiteseerPubmed.py --data_name citeseer --use_heuristic katz_close


##node2vec
python  main_gnn_CoraCiteseerPubmed.py  --data_name citeseer  --gnn_model mlp_model --cat_n2v_feat  --lr 0.01 --dropout 0.5 --l2 0 --num_layers 1  --num_layers_predictor 3 --hidden_channels 256 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

##MF
python  main_gnn_CoraCiteseerPubmed.py  --data_name citeseer  --gnn_model MF   --lr 0.001 --dropout 0.5 --l2 0 --num_layers 2  --num_layers_predictor 3 --hidden_channels 256 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

##mlp
python  main_gnn_CoraCiteseerPubmed.py  --data_name citeseer  --gnn_model mlp_model  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 1  --num_layers_predictor 3 --hidden_channels 256 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024


## gcn
python  main_gnn_CoraCiteseerPubmed.py  --data_name citeseer  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

## gat
python  main_gnn_CoraCiteseerPubmed.py  --data_name citeseer  --gnn_model GAT --lr 0.01 --dropout 0.5 --l2 0 --num_layers 1  --num_layers_predictor 3 --hidden_channels 256 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

## sage
python  main_gnn_CoraCiteseerPubmed.py  --data_name citeseer  --gnn_model SAGE --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 1  --num_layers_predictor 2 --hidden_channels 256 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

##gae
python  main_gae_CoraCiteseerPubmed.py  --data_name citeseer  --gnn_model GCN --with_loss_weight --lr 0.01 --dropout 0.1 --l2 1e-4 --num_layers 1  --num_layers_predictor 2 --hidden_channels 256 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024



##neognn
python  main_neognn_CoraCiteseerPubmed.py  --data_name citeseer  --gnn_model NeoGNN   --lr 0.01 --dropout 0.1 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 256 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

##buddy
python  main_buddy_CoraCiteseerPubmed.py --data_name citeseer --model BUDDY  --lr 0.001   --label_dropout 0.3  --feature_dropout 0.3 --l2 0 --hidden_channels 256  --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

##peg
python main_peg_CoraCiteseerPubmed.py --data_name citeseer --lr 0.001 --l2 0 --hidden_dim 256 

##ncn
python  main_ncn_CoraCiteseerPubmed.py  --dataset citeseer  --predictor cn1   --gnnlr 0.001 --prelr 0.001 --l2 1e-7   --predp 0.1 --gnndp 0.1  --mplayers 1 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10  --batch_size 1024 --xdp 0.4 --tdp 0.0 --pt 0.75 --gnnedp 0.0 --preedp 0.0  --probscale 6.5 --proboffset 4.4 --alpha 0.4 --ln --lnnn   --model puregcn  --testbs 512  --maskinput  --jk  --use_xlin  --tailact  --twolayerlin

##ncnc
python  main_ncn_CoraCiteseerPubmed.py  --dataset citeseer  --predictor incn1cn1   --gnnlr 0.001 --prelr 0.001 --l2 1e-7   --predp 0.5 --gnndp 0.5  --mplayers 1 --nnlayers 2   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10  --batch_size 1024 --xdp 0.4 --tdp 0.0 --pt 0.75 --gnnedp 0.0 --preedp 0.0  --probscale 6.5 --proboffset 4.4 --alpha 0.4 --ln --lnnn   --model puregcn  --testbs 512  --maskinput  --jk  --use_xlin  --tailact  --twolayerlin


###### for the following expensive models, it might take long time to run 10 seeds. We can run 10 seeds parallelly. For example, you can add  --runs 1 --seed 0 

## seal
python main_seal_CoraCiteseerPubmed.py  --data_name citeseer   --lr 0.01  --num_layers 3  --hidden_channels 256  --num_hops 3 --epochs 9999 --runs 1  --eval_steps 1 --kill_cnt 20 --batch_size 1024  

##nbfnet
config=../baseline_models/nbfnet/data_config/citeseer.yaml
python main_nbfnet_CoraCiteseerPubmed.py --data_name citeseer --lr 0.001 --dropout 0. --input_dim 32 --hidden_dims 32 32 32 32 32 32  --config $config   --epochs 9999 --batch_size 64  --gpus [0] 
