import os
import sys

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import s3fs



def gather_data_from_categories(bucket: str, prefix: str):
    # List of strings to iterate over
    specialities = ["AGRI", "CG", "PSA", "SOCET"]

    # Initialize an empty list to hold DataFrames
    training_dataframes, skipped_dataframes, unclassifiable_dataframes = [], [], []
    print(training_dataframes)
    # Create an S3 filesystem object
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"},
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    # Iterate over the prefixes and load Parquet files
    for speciality in tqdm(specialities):
        training_file_path = f's3://{bucket}/{prefix}/{speciality}/preprocessed/training_data_{speciality}_NAF2025.parquet'
        print(f"Loading {training_file_path}")
        skipped_file_path = f's3://{bucket}/{prefix}/{speciality}/preprocessed/skipped_data_{speciality}_NAF2025.parquet'
        print(f"Loading {skipped_file_path}")
        unclassifiable_file_path = f's3://{bucket}/{prefix}/{speciality}/preprocessed/unclassifiable_data_{speciality}_NAF2025.parquet'
        print(f"Loading {unclassifiable_file_path}")

        try:
            training_df = pd.read_parquet(training_file_path, filesystem=fs)
            skipped_df = pd.read_parquet(skipped_file_path, filesystem=fs)
            unclassifiable_df = pd.read_parquet(unclassifiable_file_path, filesystem=fs)
        except FileNotFoundError as e:
            # Handle the case where the file does not exist for now
            print(f"{e}: The file does not exist yet.")

        training_dataframes.append(training_df)
        skipped_dataframes.append(skipped_df)
        unclassifiable_dataframes.append(unclassifiable_df)

    # Concatenate all DataFrames
    combined_training_df = pd.concat(training_dataframes, ignore_index=True)
    combined_skipped_df = pd.concat(skipped_dataframes, ignore_index=True)
    combined_unclassifiable_df = pd.concat(unclassifiable_dataframes, ignore_index=True)

    # Save all the collected training set back to S3
    combined_training_df.to_parquet(f"s3://{bucket}/{prefix}/preprocessed/training_data_NAF2025.parquet", filesystem=fs)
    combined_skipped_df.to_parquet(f"s3://{bucket}/{prefix}/preprocessed/skipped_data_NAF2025.parquet", filesystem=fs)
    combined_unclassifiable_df.to_parquet(f"s3://{bucket}/{prefix}/preprocessed/unclassifiable_data_NAF2025.parquet", filesystem=fs)


if __name__ == "__main__":
    annotation_extraction_prefix = 'label-studio/annotation-campaign-2024/rev-NAF2025' # str(sys.argv[1])


    gather_data_from_categories("projet-ape", annotation_extraction_prefix)
