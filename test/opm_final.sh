#!/bin/bash
prog=potential.py
input=$1
energy=$2
isospin=$3
number=$4
version=$5
runtrig="0"
read num1 <<<${input//[^0-9]/ }
read char1 <<<${input//[0-9]/ }
DIRECTORY=.
TRUEDIR=$(cd -- "$DIRECTORY" && pwd)
level1="${TRUEDIR}/${char1}/${num1}"
path="${TRUEDIR}/${char1}/${num1}/${energy}"
template="${input}_template.inp"
plotter="wrkplot_final.py"
text="${input}.txt"
info_str="${input}${energy}${isospin}"
output="${info_str}.inp"
firstfile="${info_str}_first.ecis"
midfile="${info_str}_middle.ecis"
lastfile="${info_str}_last.ecis"
chifile="${info_str}_last.txt"
new_text="${info_str}.txt"
new_template="${info_str}_template.inp"
cp "${level1}/${text}" "${TRUEDIR}/${new_text}"
cp "${level1}/${template}" "${TRUEDIR}/${new_template}"
cd "${path}/"
find . -name "*${info_str}*" -exec cp {} "${TRUEDIR}/" \;
cd "${TRUEDIR}/"
dwba_on="on"
dwba_off1="mid"
dwba_off2="off"
echo "Thread #$number is running $input at $energy MeV"
echo "Running DWBA on -- Thread #$number"
python3 $prog $new_text $energy $new_template $dwba_on $version
echo "Running ecis -- Thread #$number"
./ecis < $output > $firstfile
cp $output "${output}.first"
python3 $plotter $firstfile
echo "Running DWBA off w/ original potential -- Thread #$number"
python3 $prog $new_text $energy $new_template $dwba_off1 $version
echo "Running ecis -- Thread #$number"
./ecis < $output > $midfile
cp $output "${output}.mid"
echo "Running DWBA off -- Thread #$number"
python3 $prog $new_text $energy $new_template $dwba_off2 $version
python3 var_switcher_final.py $lastfile $output "1" $input $new_text $energy
./ecis < $output > $lastfile
echo "Plotting States -- Thread #$number"
cp $output "${output}.last"
python3 var_switcher_final.py $lastfile $output '100' $input $new_text $energy
python3 $plotter $lastfile
rm $new_template $new_text
look_string="${info_str}"
mv -v *$look_string* "${path}/"
