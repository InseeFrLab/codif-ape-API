export MC_HOST_s3=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT
EXTRACT_DB=$(python extract_db.py "$NAMESPACE/extractions" "1512" "1")
mc mv $EXTRACT_DB "s3/projet-ape/label-studio/annotation-campaign-2024/NAF2008/data-samples/queue/"
mc cp --recursive "s3/projet-ape/label-studio/annotation-campaign-2024/NAF2008/data-samples/queue/" "s3/projet-ape/label-studio/annotation-campaign-2024/NAF2008/data-samples/archive/"
