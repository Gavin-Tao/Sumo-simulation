#!/bin/bash
cd /home/xiaowen/sumo-rl; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
C=nets/dublin/weekday_pairs50/calib; S=nets/dublin/weekday_18h_reroute_v2_samevtype; R="car_weekday_18h.rou.xml,bus_weekday_18h.rou.xml,amb_weekday_18h_probe11h.rou.xml"
run() { python experiments/tools/dublin/scenario/scenario_check.py "$@" > "${@: -1}.log" 2>&1; }
for s in 123 124 125; do
  run --policy random --seed $s --out $C/v4_random_s$s &
  run --policy fixed --green 20 --seed $s --out $C/v4_fixed20_s$s &
  wait
  [ $s = 123 ] || { run --scenario $S --sumocfg dublin_weekday_18h_probe11h_8std_splitslot.sumocfg --routes $R --policy random --seed $s --out $C/ref18h_random_s$s & run --scenario $S --sumocfg dublin_weekday_18h_probe11h_8std_splitslot.sumocfg --routes $R --policy fixed --green 20 --seed $s --out $C/ref18h_fixed20_s$s & wait; }
done
echo V4_DONE
