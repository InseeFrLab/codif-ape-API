AWS_ACCESS_KEY_ID=`vault kv get -field=ACCESS_KEY onyxia-kv/projet-ape/s3` && export AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=`vault kv get -field=SECRET_KEY onyxia-kv/projet-ape/s3` && export AWS_SECRET_ACCESS_KEY
export MC_HOST_s3=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT

LOG_FILE_PATH=log_files/log_zip

# Retrieves raw logs files from s3
mc cp -r s3/projet-ape/log_files/raw/ log_files/raw

# Create an empty directory
mkdir -p $LOG_FILE_PATH

# Move only .gz files to this new directory
mv -n log_files/raw/*.gz $LOG_FILE_PATH/

# Unzip all .gz files
gunzip $LOG_FILE_PATH/*.gz

# Navigate to the folder
cd "$LOG_FILE_PATH" || exit

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
python utils/extract_prod_logs.py $LOG_FILE_PATH
