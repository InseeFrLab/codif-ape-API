mc cp --recursive s3/projet-ape/extractions/20240212_sirene4.parquet ./
EXTRACT_DB=$(python extract-db.py "20240212_sirene4.parquet" "2500")
mc mv $EXTRACT_DB "s3/projet-ape/label-studio/annotation-campaign-2024/NAF2008/data-samples/queue/"

# mc cp --recursive "s3/projet-ape/Label Studio/Annotation APE 2024/NAF 2008/Extract manuelle/Archive annotations/" "s3/projet-ape/label-studio/annotation-campaign-2024/NAF2008/data-samples/archive/"