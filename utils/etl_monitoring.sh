# Set environment variables
export MC_HOST_s3=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT
NAMESPACE=projet-ape

LOG_FILE_PATH_S3_IN=$NAMESPACE/log_files/raw/
LOG_FILE_PATH_S3_OUT=$NAMESPACE/log_files/preprocessed/
LOG_FILE_PATH_LOCAL=log_files/log_zip

DAY_SHIFT=21
DATE_TO_LOG=$(date --date="-$DAY_SHIFT days" +%Y-%m-%d)

POD_NAME=$(kubectl get pods -l app=codification-ape-model-deployment --no-headers -o custom-columns=":metadata.name")
PATH_TO_LOGS=/api
API_PATH_LOGS=codification_ape_log_file.log

# Retrieves raw logs files from s3
mc cp -r s3/$LOG_FILE_PATH_S3_IN log_files/raw

# Create an empty directory
mkdir -p $LOG_FILE_PATH_LOCAL

# Move only .gz files to this new directory
mv -n log_files/raw/*.gz $LOG_FILE_PATH_LOCAL/

# Unzip all .gz files
gunzip $LOG_FILE_PATH_LOCAL/*.gz

# Navigate to the folder
cd "$LOG_FILE_PATH_LOCAL" || exit

# Loop through each file in the folder
for file in *.log.*; do
    if [ -f "$file" ]; then
        # Extract date from the file name
        date_part=$(echo "$file" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}')

        # Construct the new file name
        new_name="${date_part}.${file}"

        # Rename the file
        mv "$file" "$new_name"

        echo "Renamed: $file to $new_name"
    fi
done
cd ../..

# Transform and save logs
python extract_prod_logs.py $LOG_FILE_PATH_LOCAL

# Send a batch query from a daily log
python send_batch.py $LOG_FILE_PATH_S3_OUT $DATE_TO_LOG

# Copy locally log file from API pod
kubectl cp $NAMESPACE/$POD_NAME:$PATH_TO_LOGS/$API_PATH_LOGS $API_PATH_LOGS

# Transform logs in an explicit format for a given ans store parquet in s3
python transform_logs.py $API_PATH_LOGS $DATE_TO_LOG $DAY_SHIFT
