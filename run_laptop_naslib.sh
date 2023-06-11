REPO_DIR=$HOME/experiments
INIT_SCRIPT=$HOME/experiments/init-env-laptop.sh

cd $REPO_DIR
source $INIT_SCRIPT

cd $HOME/NASLib
$mpilaunch -n 2 ./run_aux_script.sh
