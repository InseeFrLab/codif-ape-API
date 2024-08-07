# install required libraries
# pip install -r requirements.txt
# sampling from extracted db
EXTRACT_DB=$(python extract-db-otm.py "$NAMESPACE/NAF-revision/extractions/one-to-many" $NUMBER_TO_ANNOTATE_NAF2025)
mc cp --recursive $DATA_SAMPLED_OTM_CG_QUEUE_PATH $DATA_SAMPLED_OTM_CG_ARCHIVE_PATH
mc cp --recursive $DATA_SAMPLED_OTM_AGRI_QUEUE_PATH $DATA_SAMPLED_OTM_AGRI_ARCHIVE_PATH
mc cp --recursive $DATA_SAMPLED_OTM_PSA_QUEUE_PATH $DATA_SAMPLED_OTM_PSA_ARCHIVE_PATH
mc cp --recursive $DATA_SAMPLED_OTM_SOCET_QUEUE_PATH $DATA_SAMPLED_OTM_SOCET_ARCHIVE_PATH
mc cp --recursive $DATA_SAMPLED_OTM_SSP_QUEUE_PATH $DATA_SAMPLED_OTM_SSP_ARCHIVE_PATH
