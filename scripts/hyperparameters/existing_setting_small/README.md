
# Hyperparameters for Cora, Citeseer, and Pubmed under the existing setting

`cora.sh`, `citeseer.sh`, `pubmed.sh` provide the parameters for  Cora, Citeseer, and Pubmed under the existing setting.  SEAL and NBFNet might need more time to finish 10 seeds, you can run one seed at a time or run 10 seeds parallelly. To run a single seed, you can add `--runs 1 --seed 0` as command line arguments. One example to run SEAL for a single seed on Cora is:
```
python main_seal_CoraCiteseerPubmed.py  --data_name cora  --lr 0.01  --num_layers 3  --hidden_channels 256  --num_hops 3 --epochs 9999  --eval_steps 1 --kill_cnt 20 --batch_size 1024 --runs 1  --seed 0
```