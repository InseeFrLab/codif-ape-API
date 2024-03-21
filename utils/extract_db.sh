mc cp --recursive s3/projet-ape/extractions/20240321_sirene4.parquet ./
EXTRACT_DB=$(python extract_db.py "20240321_sirene4.parquet" "1890")
mc mv $EXTRACT_DB "s3/projet-ape/label-studio/annotation-campaign-2024/NAF2008/data-samples/queue/"
mc cp --recursive "s3/projet-ape/label-studio/annotation-campaign-2024/NAF2008/data-samples/queue/" "s3/projet-ape/label-studio/annotation-campaign-2024/NAF2008/data-samples/archive/"
