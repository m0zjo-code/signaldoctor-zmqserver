#!/bin/bash

cd /home/jonathan/signaldoctor-zmqserver

python3 test_model_1D.py \
--weights /mnt/datastore/FYP/prototypenetworks/MeanPSD_Adamax_1_2_1527865687.h5 \
--arch /mnt/datastore/FYP/prototypenetworks/MeanPSD_Adamax_1_2_1527865687.nn \
--test /home/jonathan/signaldoctor-zmqserver/nnetsetup/MeanPSDTrainingData.npz \
--analysis /mnt/datastore/FYP/training_sets/HF_SetV4_NOISE_reduced \
--mode psd

python3 test_model_1D.py \
--weights /mnt/datastore/FYP/prototypenetworks/MaxPSD_Adamax_1_2_1527865830.h5 \
--arch /mnt/datastore/FYP/prototypenetworks/MaxPSD_Adamax_1_2_1527865830.nn \
--test /home/jonathan/signaldoctor-zmqserver/nnetsetup/MaxPSDTrainingData.npz \
--analysis /mnt/datastore/FYP/training_sets/HF_SetV4_NOISE_reduced \
--mode max_spectrum

python3 test_model_1D.py \
--weights /mnt/datastore/FYP/prototypenetworks/MinPSD_Adamax_1_2_1527865966.h5 \
--arch /mnt/datastore/FYP/prototypenetworks/MinPSD_Adamax_1_2_1527865966.nn \
--test /home/jonathan/signaldoctor-zmqserver/nnetsetup/MinPSDTrainingData.npz \
--analysis /mnt/datastore/FYP/training_sets/HF_SetV4_NOISE_reduced \
--mode min_spectrum

python3 test_model_1D.py \
--weights /mnt/datastore/FYP/prototypenetworks/VarPSD_Adamax_1_2_1527866038.h5 \
--arch /mnt/datastore/FYP/prototypenetworks/VarPSD_Adamax_1_2_1527866038.nn \
--test /home/jonathan/signaldoctor-zmqserver/nnetsetup/VarTrainingData.npz \
--analysis /mnt/datastore/FYP/training_sets/HF_SetV4_NOISE_reduced \
--mode variencespectrum


