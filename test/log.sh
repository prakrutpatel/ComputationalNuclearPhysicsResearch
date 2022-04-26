#!/bin/bash
DIRECTORY=.
TRUEDIR=$(cd -- "$DIRECTORY" && pwd)
fold=$(date +'%m.%d.%Y,%H-%M-%S');
mkdir -p  "${TRUEDIR}/run_data/${fold}"
input=$1
run_num=$2
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
   mkdir -p "${TRUEDIR}/run_data/${fold}/${char1}/${num1}/${arr[1]}/"
   cd "${path}/"
   mv *.eps "${TRUEDIR}/run_data/${fold}/${char1}/${num1}/${arr[1]}/"
   mv *.inp* "${TRUEDIR}/run_data/${fold}/${char1}/${num1}/${arr[1]}/"
   mv *.ecis "${TRUEDIR}/run_data/${fold}/${char1}/${num1}/${arr[1]}/"
   mv *.txt "${TRUEDIR}/run_data/${fold}/${char1}/${num1}/${arr[1]}/"
   mv *.betas "${TRUEDIR}/run_data/${fold}/${char1}/${num1}/${arr[1]}/"
   cd "${TRUEDIR}/"
done < "$input"
echo "Random Beta ${run_num} " >> Readme.log
#printf "Ctrl+D to stop logging\nEnter log:\n" && cat >Readme.log
mv Readme.log "${TRUEDIR}/run_data/${fold}/"
mv output.txt "${TRUEDIR}/run_data/${fold}/"
cp results_original.txt "${TRUEDIR}/run_data/${fold}/"
cp results_final.txt "${TRUEDIR}/run_data/${fold}/"
