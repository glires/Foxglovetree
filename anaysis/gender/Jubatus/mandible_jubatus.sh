#!/bin/sh
export PATH=/net/gpfs/opt/anacondap/bin:$PATH
json=config.json
port=49800
for sample in `seq 277`
do
  jubaclassifier --configpath $json --rpc-port $port > /dev/null 2> /dev/null &
  juba_pid=$!
  ./mandible_jubatus.py $port ../data/mandible_vectors.npy $sample 100
  kill $juba_pid
  let port++
done
exit
