import pandas as pd
from dateutil import parser
from pandas import json_normalize
import ast
from datetime import datetime
import sys


def upload_logs(log_file_path: str, start_time: str, end_time: str):
    # Liste pour stocker les entrées de logs
    log_entries = []

    # Convertir les chaînes de temps en objets datetime    
    start_time = parser.parse(start_time)
    end_time = parser.parse(end_time)

    # Lecture du fichier de logs
    with open(log_file_path, 'r') as file:
        for line in file:
            # Extraction de la date et de l'heure de chaque ligne de log
            log_time_str = line.split(' - ')[0].strip()
            log_time = parser.parse(log_time_str)

            # Vérifier si la ligne est dans l'intervalle de temps spécifié
            if start_time <= log_time <= end_time:
                # Ajouter la ligne de log à la liste
                log_entries.append(line)

    # Créer un DataFrame pandas à partir des entrées de logs
    columns = ['Timestamp', 'LogLevel', 'Message']
    data = []

    for entry in log_entries:
        timestamp_str, log_level, message = entry.split(' - ', 2)
        timestamp = parser.parse(timestamp_str)
        data.append([timestamp, log_level, message])

    log_df = pd.DataFrame(data, columns=columns)


    return log_df


def process_logs(logs: pd.DataFrame):

    filtered_logs = logs[(logs['LogLevel'] == 'INFO') & (logs["Message"].str.startswith("{'Query'"))]
    extracted_data = [ast.literal_eval(message) for message in filtered_logs['Message']]
    processed_logs = json_normalize(extracted_data)
    processed_logs.set_index(filtered_logs.index, inplace=True)
    return processed_logs


def main(log_file_path: str, start_time: str, end_time: str):

    logs = upload_logs(log_file_path, start_time, end_time)

    data_logs = process_logs(logs)

    df = pd.merge(logs["Timestamp"], data_logs, how='inner', left_index=True, right_index=True).reset_index(drop=True)

    df.to_parquet(f"logs_{datetime.now().strftime('%Y-%m-%d')}.parquet")


if __name__ == "__main__":

    log_file_path = str(sys.argv[1])
    start_time = str(sys.argv[2])
    end_time = str(sys.argv[3])

    main(log_file_path, start_time, end_time)
