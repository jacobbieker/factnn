#!/usr/bin/env bash

sbatch mc_xy.batch
sbatch mc_azzd.batch
sbatch vgg.batch
sbatch vgg_sep.batch
sbatch mc_sep_short.batch
sbatch mc_phitheta_short.batch
sbatch mc_holch_short.batch
sbatch mc_energy_short.batch