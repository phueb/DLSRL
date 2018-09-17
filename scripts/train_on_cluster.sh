#!/usr/bin/env bash

# kill workers
for i in bengio hawkins hebb hinton hoff lecun norman pitts;
do
    echo Killing worker on $i
    ssh $i <<- 'EOF'
        ps auxww | grep "celery worker" | awk '{print $2}' | xargs kill -9
EOF
done

# start wokers
# remove old tensorboard directories
for i in hoff norman hebb hinton pitts hawkins lecun bengio;
do
    echo Starting worker on $i
    ssh $i <<- EOF
        export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64"
        export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:/usr/local/cuda-8.0/extras/CUPTI/lib64"
        export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64"
        cd /media/lab/DLSRL
        rm -R /media/lab/DLSRL/tb/$i
        nohup celery worker -l info -A celery_app --concurrency 1 > /media/lab/DLSRL/worker_stdout/\$(hostname)_log.txt 2>&1 &
EOF
done

# submit tasks to workers
ssh s76 <<- 'EOF'
    cd /home/lab/DLSRL
    python3 celery_client.py
    nohup flower -A celery_app --port=5001 > /dev/null 2>&1 &
EOF