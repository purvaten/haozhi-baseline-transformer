# examples:
# -- billiard:
# ---- sh test_plan.sh rinv0-r2-h64w64-v2-6 oin_v1_vinp vinp_0_t20 0
# -- PHYRE
# ---- sh test_plan.sh PHYREv1 oin_v1_vinp vinp_2_cvfeat_cond_t20_40r null null hitting 1
DATASET=$1
ARCH=$2
ID=$3
GPU=$4
SID=$5
EID=$6
FID=$7
iter=best
python test.py --gpus "${GPU}" \
--cfg "outputs/phys/${DATASET}/${ID}/config.yaml" \
--predictor-arch "${ARCH}" \
--predictor-init "outputs/phys/${DATASET}/${ID}/ckpt_${iter}.path.tar" \
--eval-hit \
--start-id "${SID:=0}" \
--end-id "${EID:=25}" \
--fold-id "${FID:=0}"
