task average
  < model=$sockeye_model@train_model
  :: checkpoints=$best_checkpoints
  :: metric=$averaging_metric
  > averaged_model=averaged_model {
 
  cp -r $model averaged_model
  rm averaged_model/params.best
  python -m sockeye.average -o averaged_model/params.best -n $checkpoints --metric $metric averaged_model
  
  # do this to save space
  find averaged_model -regex '.*[0-9]$' -delete

}

task translate : sockeye 
  < model=$averaged_model@average
  < input=$spm_source_test_encoded@spm_encode
  :: beam=$beam_size 
  > translation=translation {

  python -m sockeye.translate --device-ids -1 --models $model --input $input --output translation

}
