task data
  :: clean_corpus=@
  > data=data {

  for lang in {ar,cs,de,fr}
  do
    
    echo "Downloading and extracting data for $lang"
    # download training and dev data
    wget https://wit3.fbk.eu/archive/2016-01//texts/$lang/en/$lang-en.tgz
    wget https://wit3.fbk.eu/archive/2016-01//texts/en/$lang/en-$lang.tgz

    # extract training and dev data
    tar -xf $lang-en.tgz
    tar -xf en-$lang.tgz

    # extract dev set and strip xml
    cd en-$lang
    cat IWSLT16.TED.tst*.en-$lang.$lang.xml > dev.all.$lang
    cat IWSLT16.TED.tst*.en-$lang.en.xml > dev.all.en
    grep '^<seg.*/seg>$' dev.all.$lang | sed 's/^<seg.*"> //;s/ <\/seg>$//;' > dev.$lang
    grep '^<seg.*/seg>$' dev.all.en | sed 's/^<seg.*"> //;s/ <\/seg>$//;' > dev.en
    cd ..
    
    # arrange training data and dev data
    mkdir -p data/en2$lang
    python $clean_corpus en-$lang/train.tags.en-$lang.en en-$lang/train.tags.en-$lang.$lang
    mv en-$lang/train.tags.en-$lang.en data/en2$lang/train.en
    mv en-$lang/train.tags.en-$lang.$lang data/en2$lang/train.$lang
    mv en-$lang/dev.{en,$lang} data/en2$lang/
    rm -rf en-$lang* $lang-en*

    # donwload test data
    wget https://wit3.fbk.eu/archive/2016-01-test//texts/$lang/en/$lang-en.tgz 
    wget https://wit3.fbk.eu/archive/2016-01-test//texts/en/$lang/en-$lang.tgz 

    # extract test data
    tar -xf $lang-en.tgz
    tar -xf en-$lang.tgz

    # extract dev set and strip xml
    cd en-$lang
    cat IWSLT* > all.xml
    grep '^<seg.*/seg>$' all.xml | sed 's/^<seg.*"> //;s/ <\/seg>$//;' > test.en
    
    cd ../$lang-en
    cat IWSLT* > all.xml
    grep '^<seg.*/seg>$' all.xml | sed 's/^<seg.*"> //;s/ <\/seg>$//;' > test.$lang
    cd ..

    # arrange test data
    mv en-$lang/test.en data/en2$lang/
    mv $lang-en/test.$lang data/en2$lang/
    rm -rf en-$lang* $lang-en*

    # duplicate data folder
    cp -r data/en2$lang data/${lang}2en
    
  done
}
