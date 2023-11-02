AWS_ACCESS_KEY_ID=`vault kv get -field=ACCESS_KEY onyxia-kv/projet-ape/s3` && export AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=`vault kv get -field=SECRET_KEY onyxia-kv/projet-ape/s3` && export AWS_SECRET_ACCESS_KEY
LOG_FILE_PATH=log_files/log_zip

# Retrieves raw logs files from s3
mc cp -r s3/projet-ape/log_files/raw/ log_files/raw

# Create an empty directory
mkdir -p $LOG_FILE_PATH

# Move only .gz files to this new directory
mv -n log_files/raw/*.gz $LOG_FILE_PATH/

# Unzip all .gz files
gunzip $LOG_FILE_PATH/*.gz

# Transform and save logs
python ./src/extract_logs.py $LOG_FILE_PATH
