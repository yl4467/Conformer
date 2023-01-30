# Comformer


## Get start
Dependencies should be installed using the following command before training:
pip install -r requirements.txt


For training the model,  the full command line can be seen as follows:

```bash
python -u train.py --model <model> --data <data>
--root_path <root_path> --data_path <data_path> --features <features>
--target <target> --freq <freq> --checkpoints <checkpoints>
--seq_len <seq_len> --label_len <label_len> --pred_len <pred_len>
--enc_in <enc_in> --dec_in <dec_in> --c_out <c_out> --d_model <d_model>
--n_heads <n_heads> --e_layers <e_layers> --d_layers <d_layers>
--normal_layers <normal_layers> --enc_lstm <enc_lstm> --dec_lstm <--enc_lstm> 
--weight <weight> --window <--window> --d_ff <d_ff> --padding <padding>
--distil --dropout <dropout> --attn <attn> --embed <embed> --activation <activation>
--do_predict --mix --cols <cols> --itr <itr> --num_workers <num_workers> 
--train_epochs <train_epochs> --batch_size <batch_size> --patience <patience> --des <des>
--learning_rate <learning_rate> --loss <loss> --lradj <lradj>
--use_amp --inverse --use_gpu <use_gpu> --gpu <gpu> --use_multi_gpu --devices <devices>
```
"--features" This can be set to M,S,MS (M : multivariate predict multivariate, S : univariate predict univariate, MS : multivariate predict univariate)
"--target" Target feature in S or MS task.
"-seq_len" Input sequence length of Informer encoder (defaults to 96)
"
"--itr" represents the number of repeats to run the training model, 
"--root_path" is the path to the data file, 
"--data_path" points to the name of data file. 

## Requirements

* Python 3.8
* numpy == 1.19.4
* pandas == 0.25.1
* scikit_learn == 0.21.3
* torch == 1.8.0

Use the following command to set the environment:
```bash
pip install -r requirements.txt
```

##