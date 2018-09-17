#!/usr/bin/env bash

DLSRL_DIR=/home/lab/cluster/celery/dlsrl

# copy task to s76
#echo Copying task to s76
#scp -r ../src s76:$DLSRL_DIR
#scp ../app.py s76:$DLSRL_DIR
#scp ../client.py s76:$DLSRL_DIR
#scp ../celeryconfig.py s76:$DLSRL_DIR

git push cluster

# kill workers
for i in bengio hawkins hebb hinton hoff lecun norman pitts;
do
    echo Killing worker on $i
    ssh $i <<- 'EOF'
        ps auxww | grep "celery worker" | awk '{print $2}' | xargs kill -9
EOF
done

# start workers
# remove old tensorboard directories
for i in hoff norman hebb hinton pitts hawkins lecun bengio;
do
    echo Starting worker on $i
    ssh $i <<- EOF
        export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64"
        export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:/usr/local/cuda-8.0/extras/CUPTI/lib64"
        export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64"
        rm -R /media/lab/cluster/tensorboard/dlsrl/$i
        cd /media/lab/cluster/celery/dlsrl
        nohup celery worker -l info -A app --concurrency 1 > /media/lab/cluster/logs/dlsrl/\$(hostname)_log.txt 2>&1 &
EOF
done

# submit tasks to workers
ssh s76 <<- EOF
    cd /home/lab/cluster/celery/dlsrl
    python3 client.py
    nohup flower -A app --port=5001 > /dev/null 2>&1 &
EOF