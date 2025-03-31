#!/bin/bash

# Set environment variables
NAMESPACE='projet-ape'
LOG_FILE_PATH_S3_IN=$NAMESPACE/log_files/raw/
LOG_FILE_PATH_S3_OUT=$NAMESPACE/log_files/preprocessed/

DATE_TO_LOG=$(date --date="-$DAY_SHIFT days" +%Y-%m-%d)


POD_NAME=$(kubectl get pods -l app=codification-ape-model-deployment --no-headers -o custom-columns=":metadata.name")

# Retrieves raw logs files from s3
mc cp -r s3/$LOG_FILE_PATH_S3_IN log_files/raw > /dev/null

# Create an empty directory
mkdir -p $LOG_FILE_PATH_LOCAL
mkdir -p $DATA_FILE_PATH_LOCAL

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

    fi
done
cd ../..

# Transform and save logs
#python extract_prod_logs.py $LOG_FILE_PATH_LOCAL

# Send a batch query from a daily log
#python send_batch.py $LOG_FILE_PATH_S3_OUT $DATE_TO_LOG

# Copy locally log file from API pod
#kubectl cp $NAMESPACE/$POD_NAME:$PATH_TO_LOGS/$API_PATH_LOGS $API_PATH_LOGS

# Transform logs in an explicit format for a given ans store parquet in s3
#python transform_logs.py $API_PATH_LOGS $DATE_TO_LOG $DAY_SHIFT


##### EVALUATION ON TEST SET

# Retrieve recursively all annotation data and copy locally
mc ls s3/$NAMESPACE/$PATH_ANNOTATION_RESULTS | awk '{print "s3/'$NAMESPACE'/'$PATH_ANNOTATION_RESULTS'/" $5}' | xargs -I {} mc cp --recursive {} ./$DATA_FILE_PATH_LOCAL  > /dev/null

# Transform and save labeled test data in NACE rev 2
python extract_test_data.py $DATA_FILE_PATH_LOCAL $PATH_ANNOTATION_PREPROCESSED

##### OTM EXTRACTION FOR TRAINING SET

# Transform and save labeled training data in NACE rev 2.1
# List of CATEGORY values
categories="AGRI CG PSA SOCET"
# Loop through each CATEGORY value
for CATEGORY in $categories; do
    echo "Processing for CATEGORY: $CATEGORY"
    PATH_ANNOTATION_RESULTS_NAF2025='label-studio/annotation-campaign-2024/rev-NAF2025/'$CATEGORY'/data-annotated/in-progress'
    PATH_ANNOTATION_PREPROCESSED_NAF2025='label-studio/annotation-campaign-2024/rev-NAF2025/'$CATEGORY'/preprocessed'
    # Create an empty directory
    mkdir -p $DATA_FILE_PATH_LOCAL_$CATEGORY
    # Retrieve recursively all annotation data and copy locally
    mc ls s3/$NAMESPACE/$PATH_ANNOTATION_RESULTS_NAF2025 | awk '{print "s3/'$NAMESPACE'/'$PATH_ANNOTATION_RESULTS_NAF2025'/" $5}' | xargs -I {} mc cp --recursive {} ./$DATA_FILE_PATH_LOCAL_$CATEGORY  > /dev/null
    # Transform and save annotation data for each category
    python extract-train-data-otm.py "$DATA_FILE_PATH_LOCAL_$CATEGORY" "$PATH_ANNOTATION_PREPROCESSED_NAF2025" $CATEGORY
done
# Collect final labeled training set gathered from each category
python collect-train-data-otm.py 'label-studio/annotation-campaign-2024/rev-NAF2025'

# Predict with current model to send data to dashboard
python send_batch_test_data.py $NAMESPACE/$PATH_ANNOTATION_PREPROCESSED $PATH_ANNOTATION_DASHBOARD/current-model $CURRENT_MODEL_API_PATH

# Predict with next/retrained model to send data to dashboard
python send_batch_test_data.py $NAMESPACE/$PATH_ANNOTATION_PREPROCESSED $PATH_ANNOTATION_DASHBOARD/next-model $NEXT_MODEL_API_PATH
