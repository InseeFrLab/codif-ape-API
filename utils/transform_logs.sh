NAMESPACE=projet-ape
POD_NAME=$(kubectl get pods -l app=codification-ape-model-deployment --no-headers -o custom-columns=":metadata.name")
PATH_TO_LOGS=/api
LOCAL_PATH_LOGS=codification_ape_log_file.log
START_DATE=$(date -d "yesterday 00:0:00" '+%Y-%m-%d %H:%M:%S')
END_DATE=$(date -d "yesterday 23:59:59" '+%Y-%m-%d %H:%M:%S')
export MC_HOST_s3=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT

# Copy locally log file from API pod
kubectl cp $NAMESPACE/$POD_NAME:$PATH_TO_LOGS/codification_ape_log_file.log $LOCAL_PATH_LOGS

# Transform logs in an explicit format for a given 
python transform_logs.py $LOCAL_PATH_LOGS $START_DATE $END_DATE

mc cp logs_$(date +'%Y-%m-%d').parquet s3/projet-ape/log_files/test/logs_$(date +'%Y-%m-%d').parquet
