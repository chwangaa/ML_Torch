#!/bin/bash
for i in {1..54}
do
   lokisim -run build/loki-gemm-test -PCOMPUTE_TILE_COLUMNS=2 --args $i $i $i 8
done

# for i in {64..128..16}
# do
#    lokisim -run build/loki-gemm-test -PCOMPUTE_TILE_COLUMNS=4 --args $i $i $i 8
# done