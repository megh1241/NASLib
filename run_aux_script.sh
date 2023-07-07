#!/bin/bash
CONFIG_FILE_PATH=$HOME'/NASLib/naslib/runners/nas_predictors/discrete_config.yaml'
#CONFIG_FILE_PATH=$HOME'/NASLib/naslib/runners/zc/zc_config.yaml'

if [[ $OMPI_COMM_WORLD_RANK -ge 3 ]]; then
	cd $HOME/experiments
	./cpp-store/server \
                --thallium_connection_string "tcp"\
                --num_threads 1 \
                --num_servers 1 \
                --storage_backend "map" \
                --ds_colocated 0
else
	python3 naslib/runners/nas_predictors/runner.py --config-file $CONFIG_FILE_PATH --transfer_weights
	#python3 naslib/runners/zc/runner.py --config-file $CONFIG_FILE_PATH --transfer_weights 
	#python3 naslib/runners/zc/runner.py --config-file $CONFIG_FILE_PATH
fi
