#!/bin/bash

CURDIR=data/ecb+

if [ -d $CURDIR ]
then 
rm -rf $CURDIR
fi

git clone https://github.com/cltl/ecbPlus.git ${CURDIR}/raw
unzip ${CURDIR}/raw/ECB+_LREC2014/ECB+.zip -d ${CURDIR}/raw/ECB+_LREC2014
rm ${CURDIR}/raw/ECB+_LREC2014/ECB+/.DS_Store
mkdir -p ${CURDIR}/interim
python src/data/make_dataset.py

python src/data/build_features.py