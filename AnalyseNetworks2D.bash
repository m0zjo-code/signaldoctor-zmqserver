#!/bin/bash

cd /home/jonathan/signaldoctor-zmqserver

python3 test_model_2D.py \
--weights /mnt/datastore/FYP/prototypenetworks/MagSpec_Adadelta_4_1_1527866143.h5 \
--arch /mnt/datastore/FYP/prototypenetworks/MagSpec_Adadelta_4_1_1527866143.nn \
--test /home/jonathan/signaldoctor-zmqserver/nnetsetup/MagSpecTrainingData.npz \
--analysis /mnt/datastore/FYP/training_sets/HF_SetV4_NOISE_reduced \
--mode magnitude

python3 test_model_2D.py \
--weights /mnt/datastore/FYP/prototypenetworks/FFTSpec_Adadelta_4_1_1527917966.h5 \
--arch /mnt/datastore/FYP/prototypenetworks/FFTSpec_Adadelta_4_1_1527917966.nn \
--test /home/jonathan/signaldoctor-zmqserver/nnetsetup/FFTTrainingData.npz \
--analysis /mnt/datastore/FYP/training_sets/HF_SetV4_NOISE_reduced \
--mode fft_spectrum

python3 test_model_2D.py \
--weights /mnt/datastore/FYP/prototypenetworks/CecSpec_Adadelta_4_1_1527900104.h5 \
--arch /mnt/datastore/FYP/prototypenetworks/CecSpec_Adadelta_4_1_1527900104.nn \
--test /home/jonathan/signaldoctor-zmqserver/nnetsetup/CecTrainingData.npz \
--analysis /mnt/datastore/FYP/training_sets/HF_SetV4_NOISE_reduced \
--mode corrcoef


