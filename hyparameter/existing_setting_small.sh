

########################### Cora:

## gcn
python  main_gnn_CoraCiteseerPubmed.py  --data_name cora  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

## gat
python  main_gnn_CoraCiteseerPubmed.py  --data_name cora  --gnn_model GAT --lr 0.01 --dropout 0.1 --l2 1e-7 --num_layers 1  --num_layers_predictor 3 --hidden_channels 256 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

##node2vec
# python  main_gnn_CoraCiteseerPubmed.py  --data_name cora  --gnn_model mlp_model --cat_n2v_feat  --lr 0.01 --dropout 0.1 --l2 1e-7 --num_layers 1  --num_layers_predictor 3 --hidden_channels 256 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

##neognn
python  main_neognn_CoraCiteseerPubmed.py  --data_name cora  --gnn_model NeoGNN   --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 256 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

##buddy
python  main_buddy_CoraCiteseerPubmed.py --data_name cora --model BUDDY  --lr 0.01   --label_dropout 0.1  --feature_dropout 0.1 --l2 1e-4 --hidden_channels 256  --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

##peg
python main_peg_CoraCiteseerPubmed.py --data_name cora --lr 0.001 --l2 1e-7 --hidden_dim 256 

##ncn
model=cn1
python -u main_ncn_CoraCiteseerPubmed.py  --dataset cora  --gnnlr 0.01 --prelr 0.01 --l2 1e-4  --predp 0.3 --gnndp 0.3  --mplayers 3 --nnlayers 3 --hiddim 256 --testbs 512 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024     --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4    --probscale 4.3 --proboffset 2.8 --alpha 1.0    --ln --lnnn --predictor $model  --runs 10 --model puregcn   --maskinput  --jk  --use_xlin  --tailact

##ncnc
model=incn1cn1
python -u main_ncn_CoraCiteseerPubmed.py  --dataset cora  --gnnlr 0.01 --prelr 0.01 --l2 1e-4  --predp 0.1 --gnndp 0.1  --mplayers 2 --nnlayers 1 --hiddim 128 --testbs 512 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024     --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4    --probscale 4.3 --proboffset 2.8 --alpha 1.0    --ln --lnnn --predictor $model  --runs 10 --model puregcn   --maskinput  --jk  --use_xlin  --tailact


###### for the following expensive models, it might take long time to run 10 seeds. We can run 10 seeds parallelly, just add  --runs 1 --seed 0 
##seal
python -u main_seal_CoraCiteseerPubmed.py  --data_name cora  --lr 0.01  --num_layers 3  --hidden_channels 256  --num_hops 3 --epochs 9999  --eval_steps 1 --kill_cnt 20 --batch_size 1024 

##nbfnet
config=../baseline_models/nbfnet/data_config/cora.yaml
python main_nbfnet_CoraCiteseerPubmed.py --data_name cora --lr 0.01 --dropout 0. --input_dim 32 --hidden_dims 32 32 32 32 32 32  --config $config   --epochs 9999 --batch_size 64  --gpus [0] 

