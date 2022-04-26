#!/bin/bash
prog=potential.py
input=$1
energy=$2
number=$3
isospin=$4
version=$5
runtrig="0"
read num1 <<<${input//[^0-9]/ }
read char1 <<<${input//[0-9]/ }
DIRECTORY=.
TRUEDIR=$(cd -- "$DIRECTORY" && pwd)
level1="${TRUEDIR}/${char1}/${num1}"
path="${TRUEDIR}/${char1}/${num1}/${energy}"
template="${input}_template.inp"
plotter="wrkplot.py"
text="${input}.txt"
info_str="${input}${energy}${isospin}"
info_str_no_isospin="${input}${energy}"
output="${info_str}.inp"
firstfile="${info_str}_first.ecis"
midfile="${info_str}_middle.ecis"
lastfile="${info_str}_last.ecis"
temp="${info_str}_temp.ecis"
chifile="${info_str}_last.txt"
varfile="${info_str}_var.txt"
paramfile="${info_str}_param.txt"
new_text="${info_str_no_isospin}.txt"
new_template="${info_str_no_isospin}_template.inp"
isospin_label=''
if [ "$isospin" = "n" ]; then
  isospin_label='n'
fi
if [ -f "${level1}/${text}" ] && [ ! -f "${TRUEDIR}/${new_text}" ]; then
  cp "${level1}/${text}" "${TRUEDIR}/${new_text}"
fi
if [ -f "${level1}/${template}" ] && [ ! -f "${TRUEDIR}/${new_template}" ]; then
  cp "${level1}/${template}" "${TRUEDIR}/${new_template}"
fi
cd "${path}/"
find . -name "*${input}${energy}${isospin_label}*.dat" -exec cp {} "${TRUEDIR}/" \;
cd "${TRUEDIR}/"
dwba_on="on"
dwba_off1="mid"
dwba_off2="off"
echo "Thread #$number is running $input at $energy MeV"
echo "Running DWBA on -- Thread #$number"
python3 $prog $new_text $energy $new_template $dwba_on $isospin $version
echo "Running ecis -- Thread #$number"
./ecis < $output > $firstfile
mv $output "${output}.first"
echo "Swapping variables -- Thread #$number"
python3 $plotter $firstfile $isospin $version
echo "Running DWBA off w/ original potential -- Thread #$number"
python3 $prog $new_text $energy $new_template $dwba_off1 $isospin $version
echo "Running ecis -- Thread #$number"
./ecis < $output > $midfile
mv $output "${output}.mid"
echo "Running DWBA off -- Thread #$number"
python3 $prog $new_text $energy $new_template $dwba_off2 $isospin $version
index=1
if [[ $version == 'final' ]]; then
    ./ecis < $output > $lastfile
    python3 var_switcher.py $lastfile $output $index $input $new_text $energy $isospin $version
    echo "Plotting States -- Thread #$number"
    python3 $plotter $lastfile $isospin $version
    rm "${info_str}angles.npy"
    rm "${info_str}polar.npy"
    rm "${info_str}cs.npy"
    rm "${info_str}rcs.npy"
    if [[ -f "${info_str}tcs.npy" ]];
        then
            rm "${info_str}tcs.npy"
    fi
    #rm $chifile $varfile
    look_string="${info_str}"
    mv -v *$look_string* "${path}/"
else
    for (( ; ; ))
    do
       echo "Running loop -- Thread #$number, loop index $index"
       timeout -s HUP 1h ./ecis < $output > $temp
       sig=$?
       if [ $sig -eq "0" ];
          then
             runtrig="1"
             mv $temp $lastfile
             python3 var_switcher.py $lastfile $output $index $input $new_text $energy $isospin $version
             var=$?
             if [ $var -eq "1" ];
                then
                    break
             fi
       fi
       if [ $sig -eq "124" ];
          then
             rm $temp
             break
       fi
       if [ $index -eq 100 ];
          then
             rm $temp
             echo "Problem, ran 100 times!"
             rm $temp
             runtrig="0"
             break
       fi
       ((index=index+1))
    done
    if [ $runtrig -eq "1" ];
       then
          echo "Plotting States -- Thread #$number"
          echo "Last File  #$lastfile   $isospin"
          python3 $plotter $lastfile $isospin $version
    fi
    if [ $runtrig -eq "0" ];
       then
          echo "Unable to plot"
    fi
    rm "${info_str}angles.npy"
    rm "${info_str}polar.npy"
    rm "${info_str}cs.npy"
    rm "${info_str}rcs.npy"
    if [[ -f "${info_str}tcs.npy" ]];
        then
            rm "${info_str}tcs.npy"
    fi
    #rm $chifile $varfile
    look_string="${info_str}"
    mv -v *$look_string* "${path}/"
fi
