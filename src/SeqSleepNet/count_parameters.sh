CUDA_VISIBLE_DEVICES="3,-1" python3 count_parameter_seqsleepnet.py --nchannel 1 --dropout_keep_prob_rnn 0.75 --seq_len 20 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size1 64
