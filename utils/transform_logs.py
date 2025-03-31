import ast
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
from dateutil import parser
from pandas import json_normalize


def upload_logs(log_file_path: str, day_to_retrieve: str, day_shift: str):
    # Liste pour stocker les entrées de logs
    log_entries = []

    # Convertir les chaînes de temps en objets datetime
    start_time = parser.parse(day_to_retrieve) + timedelta(days=int(day_shift))
    end_time = datetime(start_time.year, start_time.month, start_time.day, 23, 59, 59)

    # Lecture du fichier de logs
    with open(log_file_path, "r") as file:
        for line in file:
            # Extraction de la date et de l'heure de chaque ligne de log
            log_time_str = line.split(" - ")[0].strip()
            log_time = parser.parse(log_time_str)

            # Vérifier si la ligne est dans l'intervalle de temps spécifié
            if start_time <= log_time <= end_time:
                # Ajouter la ligne de log à la liste
                log_entries.append(line)

    # Créer un DataFrame pandas à partir des entrées de logs
    columns = ["Timestamp", "LogLevel", "Message"]
    data = []

    for entry in log_entries:
        timestamp_str, log_level, message = entry.split(" - ", 2)
        timestamp = parser.parse(timestamp_str) - timedelta(days=int(day_shift))
        data.append([timestamp, log_level, message])

    log_df = pd.DataFrame(data, columns=columns)

    return log_df


def process_logs(logs: pd.DataFrame):
    filtered_logs = logs[(logs["LogLevel"] == "INFO") & (logs["Message"].str.startswith("{'Query'"))]
    extracted_data = [ast.literal_eval(message) for message in filtered_logs["Message"]]
    processed_logs = json_normalize(extracted_data)
    processed_logs.set_index(filtered_logs.index, inplace=True)
    return processed_logs


def save_to_s3(table: pa.Table, bucket: str, path: str):
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"},
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    pq.write_to_dataset(
        table,
        root_path=f"s3://{bucket}/{path}",
        partition_cols=["date"],
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        filesystem=fs,
    )


def main(log_file_path: str, day_to_retrieve: str, day_shift: str):
    # Upload log file from API pod and filter by date
    logs = upload_logs(log_file_path, day_to_retrieve, day_shift)

    # Normalize logs into a dataframe
    data_logs = process_logs(logs)

    # Add timestamp to logs
    df = pd.merge(logs["Timestamp"], data_logs, how="inner", left_index=True, right_index=True).reset_index(drop=True)

    # Add date column for partitionning
    df["date"] = df["Timestamp"].dt.strftime("%Y-%m-%d")

    # Translate pd.DataFrame into pa.Table
    arrow_table = pa.Table.from_pandas(df)

    # Save logs in a partionned parquet file in s3
    save_to_s3(arrow_table, "projet-ape", "log_files/dashboard")


if __name__ == "__main__":
    log_file_path = str(sys.argv[1])
    day_to_retrieve = str(sys.argv[2])
    day_shift = str(sys.argv[3])

    main(log_file_path, day_to_retrieve, day_shift)
