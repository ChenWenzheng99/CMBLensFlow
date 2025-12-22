#!/bin/bash

rec_dir='/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MUST/rec/QE_HO_MV'

cd "${rec_dir}/ALILENS/temp/cinv_p"
rm *.pk
cd ..

cd "${rec_dir}/ALILENS/temp/cinv_t"
rm *.pk
cd ..

cd "${rec_dir}/ALILENS/temp/ivfs"
rm *.pk
cd ..

cd "${rec_dir}/ALILENS/temp/n1_dd"
rm *.pk
cd ..

cd "${rec_dir}/ALILENS/temp/nhl_dd"
rm *.pk
cd ..

cd "${rec_dir}/ALILENS/temp/qcls_dd"
rm *.pk
cd ..

cd "${rec_dir}/ALILENS/temp/qcls_ds"
rm *.pk
cd ..

cd "${rec_dir}/ALILENS/temp/qcls_ss"
rm *.pk
cd ..

cd "${rec_dir}/ALILENS/temp/qlms_dd"
rm *.pk
cd ..

cd "${rec_dir}/ALILENS/temp/qlms_ds"
rm *.pk
cd ..

cd "${rec_dir}/ALILENS/temp/qlms_ss"
rm *.pk
cd ..

cd "${rec_dir}/ALILENS/temp/qresp"
rm *.pk
cd ..
