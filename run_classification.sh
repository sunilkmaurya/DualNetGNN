#Cora
python -u node_class.py --data cora --layer 3 --w_fc2 0.0001 --w_fc1 0.0001 --dropout 0.7 --lr_fc 0.01 --layer_norm 1 --dev 0 --hidden 64 --wd_sel 1e-05 --lr_sel 0.005 --step1_iter 400
#Citeseer
python -u node_class.py --data citeseer --layer 3 --w_fc2 0.001 --w_fc1 1e-05 --dropout 0.6 --lr_fc 0.05 --layer_norm 1 --dev 0 --hidden 64 --wd_sel 1e-05 --lr_sel 0.005 --step1_iter 500
#Pubmed
python -u node_class.py --data pubmed --layer 3 --w_fc2 0.001 --w_fc1 1e-05 --dropout 0.6 --lr_fc 0.05 --layer_norm 1 --dev 0 --hidden 64 --wd_sel 1e-05 --lr_sel 0.01 --step1_iter 400
#Chameleon
python -u node_class.py --data chameleon --layer 3 --w_fc2 0.0 --w_fc1 0.0 --dropout 0.6 --lr_fc 0.001 --layer_norm 1 --dev 0 --hidden 64 --wd_sel 1e-06 --lr_sel 0.01 --step1_iter 600
#Wisconsin
python -u node_class.py --data wisconsin --layer 3 --w_fc2 0.0001 --w_fc1 0.0001 --dropout 0.5 --lr_fc 0.05 --layer_norm 1 --dev 0 --hidden 64 --wd_sel 0.0001 --lr_sel 0.01 --step1_iter 100
#Texas
python -u node_class.py --data texas --layer 3 --w_fc2 0.0001 --w_fc1 0.0001 --dropout 0.7 --lr_fc 0.05 --layer_norm 1 --dev 0 --hidden 64 --wd_sel 0.0001 --lr_sel 0.005 --step1_iter 600
#Cornell
python -u node_class.py --data cornell --layer 3 --w_fc2 0.0 --w_fc1 0.0001 --dropout 0.5 --lr_fc 0.05 --layer_norm 1 --dev 0 --hidden 64 --wd_sel 0.0001 --lr_sel 0.005 --step1_iter 600
#Squirrel 
python -u node_class.py --data squirrel --layer 3 --w_fc2 0.0 --w_fc1 0.0001 --dropout 0.5 --lr_fc 0.001 --layer_norm 1 --dev 0 --hidden 64 --wd_sel 1e-06 --lr_sel 0.01 --step1_iter 600
#Actor
python -u node_class.py --data film --layer 3 --w_fc2 1e-05 --w_fc1 0.001 --dropout 0.7 --lr_fc 0.05 --layer_norm 1 --dev 0 --hidden 64 --wd_sel 0.0001 --lr_sel 0.01 --step1_iter 600
