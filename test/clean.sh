#!/bin/bash
DIRECTORY=.
TRUEDIR=$(cd -- "$DIRECTORY" && pwd)
input=$1
stop="end"
while IFS= read -r line
do
   if [ "${line}" = "${stop}" ];
      then
         break
   fi
   arr=($line)
   read num1 <<<${arr[0]//[^0-9]/ }
   read char1 <<<${arr[0]//[0-9]/ }
   path="${TRUEDIR}/${char1}/${num1}/${arr[1]}"
   cd "${path}/"
   rm *.eps
   rm *.inp*
   rm *.ecis
   rm *.txt
   rm *.betas
   cd "${TRUEDIR}/"
done < "$input"