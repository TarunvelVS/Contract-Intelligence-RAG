grep "ERROR" log.txt

awk '{print $2}' log.txt

grep "ERROR" log.txt | awk '{print $2}'

sort log.txt

sort log.txt | uniq -c | sort -rn

grep "ERROR" log.txt | awk '{print $2}' | sort | uniq -c | sort -rn


sed 'd/'ERROR'/'CRITICAL'/g' log.txt

cut -d',' -f2 log.txt

grep "Engineering" log.txt | awk '{print $1 $3}'

grep "Engineering" log.txt | awk -f',' '{print $1 $3}'

grep "Engineering" log.txt | awk -F',' '{print $1, $2}'


setup_env(){
    project =$1
    if [-z "$project"]; then
    echo "No Project Found"
    return 1
    fi

    mkdir -p $project
    cd $project
    python3 -m venv myenv
    source myenv/bin/activate

    if [! -f "requiremnts.txt"]; then 
    echo "No requirements.txt found"
    return 1
    fi
    pip install -r requirements.txt
}


setup_env $1
if [$? -ne 0]; then
echo "Setup failed"
exit 1
fi

echo "Setup Complete"