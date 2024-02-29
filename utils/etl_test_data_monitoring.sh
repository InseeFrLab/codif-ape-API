#Set env variables
export DATA_FILE_PATH_LOCAL=data

# Retrieve recursively all annotation data and copy locally
mc ls s3/$S3_BUCKET/$PATH_ANNOTATION_RESULTS | awk '{print "s3/'$S3_BUCKET'/'$PATH_ANNOTATION_RESULTS'/" $5}' | xargs -I {} mc cp --recursive {} ./$DATA_FILE_PATH_LOCAL

# Transform and save annotation data
python extract_test_data.py $DATA_FILE_PATH_LOCAL $PATH_ANNOTATION_PREPROCESSED

# Predict with current model to send data to dashboard
python send_batch_dashboard.py $S3_BUCKET/$PATH_ANNOTATION_PREPROCESSED $PATH_ANNOTATION_DASHBOARD/current-model $CURRENT_MODEL_API_PATH

# Predict with next/retrained model to send data to dashboard
python send_batch_dashboard.py $S3_BUCKET/$PATH_ANNOTATION_PREPROCESSED $PATH_ANNOTATION_DASHBOARD/next-model $NEXT_MODEL_API_PATH