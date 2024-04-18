export MC_HOST_s3=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT

# install required libraries
pip install -r requirements.txt

EXTRACT_DB=$(python extract_db.py "$NAMESPACE/extractions" $NUMBER_TO_ANNOTATE $MONTH_INTERVAL)
echo $DATA_SAMPLED_QUEUE_PATH
mc mv $EXTRACT_DB $DATA_SAMPLED_QUEUE_PATH
mc cp --recursive $DATA_SAMPLED_QUEUE_PATH $DATA_SAMPLED_ARCHIVE_PATH
