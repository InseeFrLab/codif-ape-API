NAMESPACE=projet-ape
LOG_FILE_PATH=$NAMESPACE/log_files/preprocessed/
DATE_TO_LOG=$(date --date="-49days" +%Y-%m-%d)

# Send a batch query from a daily log
python send_batch.py $LOG_FILE_PATH $DATE_TO_LOG
