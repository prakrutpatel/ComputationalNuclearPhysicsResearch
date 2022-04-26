#!/bin/bash
input=$1
stop="end"
while IFS= read -r line
do
   if [ "${line}" = "${stop}" ];
      then
         break
   fi
   arr=($line)
   echo $arr
   echo "${arr[0]}"
   echo "${arr[1]}"
   echo "${arr[2]}"
done < "$input"