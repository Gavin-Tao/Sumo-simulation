#!/bin/bash
cd /home/xiaowen/sumo-rl; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
S=nets/dublin/weekday_18h_15; C=$S/calib; N=nets/dublin/dublin_8std_splitslot15.net.xml; M=nets/dublin/dublin_8std_splitslot15_meta.json; R="$S/car_weekday_18h.rou.xml,$S/bus_weekday_18h.rou.xml,$S/amb_weekday_18h_probe11h.rou.xml,$S/added_flows.rou.xml"
run() { python experiments/tools/dublin/scenario/scenario_check.py --net $N --meta $M --sumocfg $S/dublin_weekday_18h_15.sumocfg --routes $R --exp-types bus,ambulance "$@" > "${@: -1}.log" 2>&1; }
for s in 123 124 125; do run --policy random --seed $s --out $C/v1_random_s$s & done; wait
for s in 123 124 125; do run --policy fixed --green 20 --seed $s --out $C/v1_fixed20_s$s & done; wait
echo V1_DONE
