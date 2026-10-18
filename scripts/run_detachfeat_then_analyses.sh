#!/bin/bash
# detach-features run, then small_analyze on it and on the four normalizer runs.
cd ${DISC_ROOT:-/workspace/disc}/scripts
./run_stride14_featrkd_soft_DETACHFEAT.sh || echo "FAILED: detachfeat"
for SUF in featrkd_soft_dist0.5_ang0_detachfeat \
           featrkd_soft_dist0.5_ang0_mumax featrkd_soft_dist0.85_ang0_mumax \
           featrkd_soft_dist0.5_ang0_mup95 featrkd_soft_dist0.8_ang0_mup95; do
  echo "########## small_analyze: $SUF ##########"
  ./small_analyze.sh "$SUF" "$SUF" > ${DISC_WS:-/workspace}/logs/small_analyze_${SUF}.log 2>&1 || echo "FAILED: $SUF"
  grep -E "coverage \(read T=1\)" ${DISC_WS:-/workspace}/logs/small_analyze_${SUF}.log || true
done
echo "########## ALL DONE ##########"
