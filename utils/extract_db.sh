# install required libraries
pip install -r requirements.txt

EXTRACT_DB=$(python extract_db.py "$NAMESPACE/extractions" $NUMBER_TO_ANNOTATE $MONTH_INTERVAL)
echo $DATA_SAMPLED_QUEUE_PATH
mc mv $EXTRACT_DB $DATA_SAMPLED_QUEUE_PATH
mc cp --recursive $DATA_SAMPLED_QUEUE_PATH $DATA_SAMPLED_ARCHIVE_PATH
