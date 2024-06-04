mc cp --recursive s3/projet-ape/NAF-revision/extractions/one-to-many/20240514_multivoque_sirene4_destinat_SOCET.parquet ./
EXTRACT_DB=$(python extract-db.py "20240514_multivoque_sirene4_destinat_SOCET.parquet" "2000")
mc mv $EXTRACT_DB "s3/nrandriamanana/label-studio/annotation-campaign-2024/rev-NAF2025/SOCET/data-samples/queue/"

# mc cp --recursive "s3/projet-ape/Label Studio/Annotation APE 2024/NAF 2008/Extract manuelle/Archive annotations/" "s3/projet-ape/label-studio/annotation-campaign-2024/NAF2008/data-samples/archive/"