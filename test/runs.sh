#!/bin/bash
for i in 1 2 3 4 5 6 7 8 9 10
do
   nohup nice python3 multi.py all.inp initial > output.txt &
   wait
   ./log.sh test.inp $i
done
