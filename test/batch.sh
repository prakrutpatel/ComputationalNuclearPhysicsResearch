input_file=$1
test="end"
while IFS= read -r line
do
        if [ "$line" = "$test" ];
                then
                        break
        fi
        ./opm.sh ${line}
done < $input_file
