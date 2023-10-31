NAMESPACE=user-tfaria
POD_NAME=$(kubectl get pods -l app=codification-ape-model-deployment --no-headers -o custom-columns=":metadata.name")
PATH_TO_LOGS=/api
LOCAL_PATH_LOGS=codification_ape_log_file.log
START_DATE=$(date +'%Y-%m-%d 00:00:00')
END_DATE=$(date +'%Y-%m-%d %H:%M:%S')

kubectl cp $NAMESPACE/$POD_NAME:$PATH_TO_LOGS/codification_ape_log_file.log $LOCAL_PATH_LOGS

python transform_logs.py $LOCAL_PATH_LOGS $START_DATE $END_DATE

mc cp logs_$(date +'%Y-%m-%d').parquet s3/projet-ape/log_files/test/logs_$(date +'%Y-%m-%d').parquet
