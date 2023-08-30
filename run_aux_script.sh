#!/bin/bash
CONFIG_FILE_PATH=$HOME'/NASLib/naslib/runners/nas_predictors/discrete_config.yaml'
#CONFIG_FILE_PATH=$HOME'/NASLib/naslib/runners/zc/zc_config.yaml'
EVAL_METHOD='run_query_zc_method'
if [[ $OMPI_COMM_WORLD_RANK -ge 2 ]]; then
	cd $HOME/experiments
	./cpp-store/server \
                --thallium_connection_string "tcp"\
		--num_threads 1 \
		--num_servers 2 \
                --storage_backend "map" \
                --ds_colocated 0
else
	python3 naslib/runners/nas_predictors/runner.py --eval_method $EVAL_METHOD --transfer_method "datastates" --data_dir "/home/ubuntu/" --config-file $CONFIG_FILE_PATH --transfer_weights
	#python3 naslib/runners/zc/runner.py --config-file $CONFIG_FILE_PATH --transfer_weights 
	#python3 naslib/runners/zc/runner.py --config-file $CONFIG_FILE_PATH
fi
