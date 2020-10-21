# DATASET=oinv9-r2-h64w64-v2-6
# DATASET=ytbv2
# DATASET=PHYREv0
# examples:
# -- billiard:
# ---- sh test_pred.sh rinv0-r2-h64w64-v2-6 oin_v1_vinp vinp_3_t20_cvfeat_cond_rerun2 0
# -- shapestack:
# ---- sh test_pred.sh shape-stack-v0 soin_v1_vinp svinp_0_t20 0
DATASET=$1
ARCH=$2
ID=$3
GPU=$4
NPLOT=$5
iter=best
python test.py \
--gpus "${GPU}" \
--cfg "outputs/phys/${DATASET}/${ID}/config.yaml" \
--predictor-arch "${ARCH}" \
--predictor-init "outputs/phys/${DATASET}/${ID}/ckpt_${iter}.path.tar" \
--plot-image "${NPLOT:=0}"
