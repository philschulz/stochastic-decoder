task spm_train : spm
  < data=$data@data
  > spm_model=spm_model
  :: source_l=$source_language
  :: target_l=$target_language
  :: num_words=$spm_vocab_size
  :: model=$spm_model_name {
  cat $data/${source_l}2${target_l}/train.${source_l} $data/${target_l}2${source_l}/train.${target_l} > concat_corpus
  $spm/src/spm_train --input=concat_corpus --vocab_size=$num_words --model_prefix=$model # --input_sentence_size 10M
  rm concat_corpus
  mv $model.model spm_model
}

task spm_encode : spm 
  < model=$spm_model@spm_train
  < data=$data@data
  :: format=$spm_format
  :: source_l=$source_language
  :: target_l=$target_language
  > spm_source_train_encoded=training.bpe.source
  > spm_source_dev_encoded=dev.bpe.source
  > spm_target_train_encoded=training.bpe.target
  > spm_target_dev_encoded=dev.bpe.target
  > spm_source_test_encoded=test.bpe.source {
  $spm/src/spm_encode --model=$model --output_format=$format < $data/${source_l}2${target_l}/train.${source_l} > training.bpe.source
  $spm/src/spm_encode --model=$model --output_format=$format < $data/${source_l}2${target_l}/dev.${source_l} > dev.bpe.source
  $spm/src/spm_encode --model=$model --output_format=$format < $data/${target_l}2${source_l}/train.${target_l} > training.bpe.target
  $spm/src/spm_encode --model=$model --output_format=$format < $data/${target_l}2${source_l}/dev.${target_l} > dev.bpe.target
  $spm/src/spm_encode --model=$model --output_format=$format < $data/${source_l}2${target_l}/test.${source_l} > test.bpe.source 
}

task spm_decode : spm
  < model=$spm_model@spm_train
  < input=$translation@translate
  :: format=$spm_format
  > plain_text_translation=plain_text_translation {
  
  $spm/src/spm_decode --model=$model --input_format=$format < $input > plain_text_translation

}
