

########################### ogbl-collab:

##ncn
model=cn1
python -u main_ncn_collabppa.py  --dataset collab  --gnnlr  0.001  --prelr 0.001 --predp 0.3 --gnndp  0.3  --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0   --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 9999 --kill_cnt 20  --batch_size 65536  --ln --lnnn --predictor $model  --model gcn  --testbs 131072  --maskinput --use_valedges_as_input   --res  --use_xlin  --tailact  


########################### ogbl-ddi:
# lr=0.001
# dropout=0.
# seed=9
# device=2
# model=cn1
# output=output/ddi/ncn/
# python -u main_ncn_ddi.py --runs 1 --seed ${seed}  --dataset ddi --device ${device} --xdp 0.05 --tdp 0.0 --gnnedp 0.0 --preedp 0.0  --gnnlr ${lr} --prelr  ${lr} --predp ${dropout} --gnndp ${dropout} --mplayers 3 --hiddim 256 --epochs 9999 --kill_cnt 20  --batch_size 65536  --ln --lnnn --predictor $model  --model puresum   --testbs 131072   --use_xlin  --twolayerlin  --res  --maskinput



########################### ogbl-citation2
# lr=0.001
# dropout=0.
# seed=9
# device=4
# model=cn1
# output=output/citation2/ncn/allmetric
# python -u main_ncn_citation2.py --runs 1 --save --seed ${seed} --dataset citation2 --device ${device} --xdp 0.0 --tdp 0.3 --gnnedp 0.0 --preedp 0.0  --gnnlr ${lr} --prelr ${lr}  --predp ${dropout} --gnndp ${dropout}  --mplayers 3 --hiddim 128 --epochs 20 --kill_cnt 20 --batch_size 65536 --ln --lnnn --predictor $model --model puregcn --res --testbs 65536 --use_xlin --tailact --proboffset 4.7 --probscale 7.0 --pt 0.3 --trndeg 128 --tstdeg 128 


