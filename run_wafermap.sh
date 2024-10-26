export EXP_DIR=./results_wafermap
export N_STEPS=1000
export RUN_NAME=runs
export PRIOR_TYPE=f_phi_prior
export CAT_F_PHI=_cat_f_phi
export F_PHI_TYPE=f_phi_supervised  #f_phi_self_supervised
export MODEL_VERSION_DIR=conditional_results/${N_STEPS}steps/nn/${RUN_NAME}/${PRIOR_TYPE}${CAT_F_PHI}/${F_PHI_TYPE}
export LOSS=conditional
export TASK=wafermap
export N_SPLITS=1
export DEVICE_ID=0
export N_THREADS=0


python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config configs/${TASK}.yml --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS}
# 
# python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config $EXP_DIR/${MODEL_VERSION_DIR}/logs/ --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS} --test --eval_best
