#!/bin/zsh

echo "HOSTNAME=$(hostname)"

eval "$(micromamba shell hook --shell zsh)"
MAMBA_CMD="micromamba activate me0segreco-py311"
echo ${MAMBA_CMD}
eval ${MAMBA_CMD}

export PYTHONPATH="${PYTHONPATH}:/users/yeonju/fpga/me0segreco/src"

TRAIN_CODE=$1
LOG_DIR=$2
CKPT_PATH=$3

echo "TRAIN_CODE=${TRAIN_CODE}"
echo "LOG_DIR=${LOG_DIR}"
echo "CHECKPOINT_PATH=${CKPT_PATH}"

RUN_CMD="python ${TRAIN_CODE} \
	--config config.yaml \
	--log_dir ${LOG_DIR}"

if [ -e ${CKPT_PATH} ]; then
	RUN_CMD=${RUN_CMD}" \
		--ckpt_path ${CKPT_PATH}"
else
	RUN_CMD=${RUN_CMD}" \
		--trainer.callbacks+=TQDMProgressBar \
		--trainer.callbacks.refresh_rate=0"
fi

echo ${RUN_CMD}
eval ${RUN_CMD}

EXIT_CODE=$?
if [[ "${EXIT_CODE}" != "0" ]]; then
#	rm -vrf lightning_logs
	echo "ERROR HAPPENED - CODE IS ${EXIT_CODE}, HOST IS $(hostname)"
	exit
fi
