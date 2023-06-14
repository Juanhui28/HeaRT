**Environment**

For PEG:

python3.7 + torch 1.8.1 (cuda10.2) + torch_geometric 1.7.2 + ogb1.3.6


Else:

The following two environments both work:
python3.9 + torch 1.12.1 (cuda11.6) + torch_geometric 2.2.0 + ogb1.3.5

python3.10 + torch 1.13.1 (cuda11.7) + torch_geometric 2.2.0 + ogb1.3.5


**Prepare Datasets**

```
python ogbdataset.py
```

**Reproduce Results**



To reproduce the results, please refer to the settings in **hyparameter** directory

example to run cora under the existing setting for cora, citeseer, pubmed
```
cd existing_setting_small

python  main_gnn_CoraCiteseerPubmed.py  --data_name cora  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024
```


<!-- example to run cora under the existing setting for ogbl-collab, ogbl-ppa, ogbl-citation2 -->
<!-- ```
python NeighborOverlap.py   --xdp 0.4 --tdp 0.0 --pt 0.75 --gnnedp 0.0 --preedp 0.0 --predp 0.55 --gnndp 0.75  --probscale 6.5 --proboffset 4.4 --alpha 0.4  --gnnlr 0.0085 --prelr 0.0078  --batch_size 384  --ln --lnnn --predictor $model --dataset Citeseer  --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1  --testbs 4096  --maskinput  --jk  --use_xlin  --tailact  --twolayerlin
``` -->



