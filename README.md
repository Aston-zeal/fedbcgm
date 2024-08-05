# fedbcgm

for fedbcgm<br>
python train_fedbcgm.py --flmethod fedbcgm --dataset mnist --model ConvNet --local_ep1 2 --epochs 30 --lr1 0.01 --gama 6 --num_users 80 --frac 0.3

for fedbcgm_group<br>
python train_fedbcgm.py --flmethod fedbcgm --dataset mnist --model ConvNet --local_ep1 2 --epochs 30 --lr1 0.01 --gama 6 --num_users 80 --frac 0.3

for fedavg<br>
python train_fedbcgm.py --flmethod fedavg --dataset mnist --model ConvNet --epochs 30 --lr1 0.01 --num_users 80 --frac 0.3

for fedprox<br>
python train_fedbcgm.py --flmethod fedprox --dataset mnist --model ConvNet --epochs 30 --lr1 0.01 --num_users 80 --frac 0.3
