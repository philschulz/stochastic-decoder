task train_model
  < training_source=$spm_source_train_encoded@spm_encode
  < training_target=$spm_target_train_encoded@spm_encode
  < dev_source=$spm_source_dev_encoded@spm_encode
  < dev_target=$spm_target_dev_encoded@spm_encode
  :: rnn_hidden=$rnn_num_hidden
  :: att_hidden=$attention_num_hidden
  :: att_type=$attention_type
  :: cell_type=$rnn_cell
  :: device_ids=$devices
  :: sequence_length=$max_sequence_length
  :: num_layers=$rnn_layers
  :: num_embed=$embedding_size
  :: opt_metric=@
  :: num_words=$vocab_size
  :: weight_decay=$l2_regularisation
  :: min_count=$word_min_count
  :: bleu=$monitor_bleu
  :: batch=$batch_size
  :: clipping=$gradient_clipping
  :: optimizer=@
  :: loss=$optimizer_loss
  :: max_num_checkpoints=$max_num_checkpoints_not_improved
  :: dropout=@
  :: latent_dim=@
  :: add_args=$additional_arguments
  :: lock_dir=@
  > sockeye_model=model {
  python -m sockeye.train -s $training_source -t $training_target --device-ids $device_ids --rnn-num-hidden $rnn_hidden --rnn-attention-num-hidden $att_hidden --rnn-attention-type $att_type \
  -o model --num-words $num_words --word-min-count $min_count --num-layers $num_layers --rnn-cell-type $cell_type --num-embed $num_embed --max-seq-len $sequence_length \
  --weight-decay $weight_decay --loss $loss --decode-and-evaluate $bleu -b $batch --clip-gradient $clipping --optimizer $optimizer --optimized-metric $opt_metric --use-tensorboard \
  --validation-source $dev_source --validation-target $dev_target --max-num-checkpoint-not-improved $max_num_checkpoints --latent-dim $latent_dim --rnn-dropout-inputs $dropout --rnn-dropout-states $dropout \
  $add_args --lock-dir $lock_dir
}
