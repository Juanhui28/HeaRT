
########################### the parameter for citeseer under HeaRT:

## CN
python main_heuristic_CoraCiteseerPubmed.py --data_name pubmed --use_heuristic CN

## AA
python main_heuristic_CoraCiteseerPubmed.py --data_name pubmed --use_heuristic AA

## RA
python main_heuristic_CoraCiteseerPubmed.py --data_name pubmed --use_heuristic RA

## shortest path
python main_heuristic_CoraCiteseerPubmed.py --data_name pubmed --use_heuristic shortest_path

## katz
python main_heuristic_CoraCiteseerPubmed.py --data_name pubmed --use_heuristic katz_close

##node2vec
python main_gnn_CoraCiteseerPubmed.py  --cat_n2v_feat --data_name pubmed  --gnn_model mlp_model --lr 0.001 --dropout 0.1 --l2 1e-7   --num_layers 3 --hidden_channels 256  --num_layers_predictor 3 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

## MF
python main_gnn_CoraCiteseerPubmed.py --data_name pubmed  --gnn_model MF --lr 0.01 --dropout 0.5 --l2 0  --num_layers 2 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 

## mlp
python main_gnn_CoraCiteseerPubmed.py  --data_name pubmed  --gnn_model mlp_model  --lr 0.001 --dropout 0.1 --l2 0 --num_layers 3 --hidden_channels 256  --num_layers_predictor 3  --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 

## gcn
python main_gnn_CoraCiteseerPubmed.py  --data_name pubmed  --gnn_model GCN  --lr 0.01 --dropout 0.3 --l2 0 --num_layers 1 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 

## gat
python main_gnn_CoraCiteseerPubmed.py  --data_name pubmed  --gnn_model GAT  --lr 0.01 --dropout 0.1 --l2 0 --num_layers 1 --hidden_channels 256  --num_layers_predictor 3  --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 

## SAGE
python main_gnn_CoraCiteseerPubmed.py  --data_name pubmed  --gnn_model SAGE  --lr 0.001 --dropout 0.1 --l2 0 --num_layers 2 --hidden_channels 256  --num_layers_predictor 3  --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 

## gae
python main_gae_CoraCiteseerPubmed.py --data_name pubmed --gnn_model GCN  --lr 0.001 --dropout 0.3 --l2 1e-4  --num_layers 2 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

## neognn
python main_neognn_CoraCiteseerPubmed.py  --data_name pubmed --lr 0.001 --dropout 0.1 --l2 1e-7   --num_layers 1 --hidden_channels 256 --num_layers_predictor 3 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

## buddy
python main_buddy_CoraCiteseerPubmed.py --model BUDDY  --data_name pubmed  --lr 0.001 --l2 1e-7  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256 --max_hash_hops 3  --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024

## peg
python main_peg_CoraCiteseerPubmed.py --data_name pubmed  --lr 0.001 --l2 0  --hidden_dim 256   --eval_steps 5  --kill_cnt 10 --epochs 9999   --batch_size 1024 

## ncn
python main_ncn_CoraCiteseerPubmed.py --predictor cn1  --testbs 512 --dataset pubmed   --gnnlr 0.001 --prelr 0.001 --l2 1e-7    --predp 0.1 --gnndp 0.1  --mplayers 2 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10  --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn  --model puregcn    --maskinput  --jk  --use_xlin  --tailact

## ncnc
python main_ncn_CoraCiteseerPubmed.py --predictor incn1cn1  --testbs 512 --dataset pubmed   --gnnlr 0.001 --prelr 0.001 --l2 0    --predp 0.1 --gnndp 0.1  --mplayers 1 --nnlayers 1   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10  --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn  --model puregcn    --maskinput  --jk  --use_xlin  --tailact

## seal
python main_seal_CoraCiteseerPubmed.py  --data_name pubmed --batch_size 512  --test_bs 512  --dynamic_train  --dynamic_val --dynamic_test --num_hops 3  --lr 0.01  --num_layers 3  --use_feature --hidden_channels 256 --epochs 15   --eval_steps 1 --kill_cnt 20 