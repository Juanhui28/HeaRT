
########################### the parameter for cora under HeaRT:

## CN
python main_heuristic_CoraCiteseerPubmed.py --data_name cora --use_heuristic CN

## AA
python main_heuristic_CoraCiteseerPubmed.py --data_name cora --use_heuristic AA

## RA
python main_heuristic_CoraCiteseerPubmed.py --data_name cora --use_heuristic RA

## shortest path
python main_heuristic_CoraCiteseerPubmed.py --data_name cora --use_heuristic shortest_path

## katz
python main_heuristic_CoraCiteseerPubmed.py --data_name cora --use_heuristic katz_close


##node2vec
python main_gnn_CoraCiteseerPubmed.py  --cat_n2v_feat  --data_name cora --gnn_model mlp_model --lr 0.01 --dropout 0.1 --l2 1e-7  --num_layers 1 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 

## MF
python main_gnn_CoraCiteseerPubmed.py  --data_name cora  --gnn_model MF  --lr 0.01 --dropout 0.5 --l2 1e-4 --num_layers 2 --hidden_channels 128  --num_layers_predictor 1 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

## mlp
python main_gnn_CoraCiteseerPubmed.py  --data_name cora  --gnn_model mlp_model  --lr 0.01 --dropout 0.1 --l2 1e-7 --num_layers 1 --hidden_channels 256  --num_layers_predictor 3  --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 

## gcn
python main_gnn_CoraCiteseerPubmed.py  --data_name cora  --gnn_model GCN  --lr 0.001 --dropout 0.5 --l2 0 --num_layers 1 --hidden_channels 256  --num_layers_predictor 3  --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 

## gat
python main_gnn_CoraCiteseerPubmed.py  --data_name cora  --gnn_model GAT  --lr 0.01 --dropout 0.5 --l2 0 --num_layers 1 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 

## SAGE
python main_gnn_CoraCiteseerPubmed.py  --data_name cora  --gnn_model SAGE  --lr 0.01 --dropout 0.1 --l2 0 --num_layers 1 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 

## gae
python main_gae_CoraCiteseerPubmed.py --data_name cora --gnn_model GCN  --lr 0.01 --dropout 0.1 --l2 1e-7  --num_layers 1 --hidden_channels 256  --num_layers_predictor 1  --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

## neognn
python main_neognn_CoraCiteseerPubmed.py  --data_name cora --lr 0.001 --dropout 0.3 --l2 0   --num_layers 1 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

## buddy
python main_buddy_CoraCiteseerPubmed.py --model BUDDY --data_name cora  --lr 0.01 --l2 1e-4  --label_dropout 0.1 --feature_dropout 0.1 --hidden_channels 256 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

## peg
python main_peg_CoraCiteseerPubmed.py --data_name cora  --lr 0.001 --l2 0  --hidden_dim 256   --eval_steps 5  --kill_cnt 10 --epochs 9999   --batch_size 1024 

## ncn
python main_ncn_CoraCiteseerPubmed.py --predictor cn1 --testbs 512  --gnnlr 0.01  --prelr 0.01 --l2 1e-7  --predp 0.1 --gnndp 0.1 --dataset cora --mplayers 3 --nnlayers 2 --hiddim 256 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0  --ln --lnnn   --model puregcn   --maskinput  --jk  --use_xlin  --tailact --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 

## ncnc
python main_ncn_CoraCiteseerPubmed.py --predictor incn1cn1 --testbs 512  --gnnlr 0.01  --prelr 0.01 --l2 0  --predp 0.1 --gnndp 0.1 --dataset cora --mplayers 2 --nnlayers 2 --hiddim 256 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0  --ln --lnnn   --model puregcn   --maskinput  --jk  --use_xlin  --tailact --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 

## seal
python main_seal_CoraCiteseerPubmed.py  --data_name cora --use_feature --dynamic_val --dynamic_test --num_hops 3  --lr 0.01  --num_layers 3   --hidden_channels 256 --epochs 9999  --eval_steps 1 --kill_cnt 20 --batch_size 1024 

## nbfnet
config=../baseline_models/nbfnet/data_config/cora.yaml
python main_nbfnet_CoraCiteseerPubmed.py --data_name cora --lr 0.01 --dropout 0.1 --input_dim 32 --hidden_dims 32 32 32 32 32 32  --config $config   --num_epoch 9999 --batch_size 64  --gpus [0] 

