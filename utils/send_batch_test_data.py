import os
import sys
from urllib.parse import urlencode

import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import s3fs

from extract_test_data_monitoring import save_to_s3


def query_batch_api(
    data: pd.DataFrame,
    api_path: str,
    nb_echos_max: int = 5,
    prob_min: float = 0.01,
):
    base_url = api_path
    params = {"nb_echos_max": nb_echos_max, "prob_min": prob_min}
    url = f"{base_url}?{urlencode(params)}"

    # Create the request body as a dictionary from the DataFrame
    request_body = data.to_dict(orient="list")
    response = requests.post(url, json=request_body)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 400:
        print(response.json()["detail"])
    else:
        print("Error occurred while querying the API.")
        return None


def reclassify_surface(d: str):
    try:
        if int(d) < 120:
            return "1"
        elif 121 <= int(d) <= 399:
            return "2"
        elif 400 <= int(d) <= 2499:
            return "3"
        else:
            return "4"
    except ValueError:
        return "null"


def get_filesystem():
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"},
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    return fs


def format_query(
    df: pd.DataFrame,
):
    subset = df.copy()
    subset["surface"] = subset["surface"].apply(reclassify_surface)
    return subset[["text_description", "type_", "nature", "surface", "event"]]


def add_prediction_columns(df, results):
    # List to store dictionaries of values for each row
    predictions = []

    # Iterate over the results for each row of the DataFrame
    for result in results:
        # Dictionary of values for this row
        row_values = {}

        # Iterate over the response items for this row
        for i in range(1, 6):
            if str(i) in result:
                response = result[str(i)]
                row_values[f"Response.{i}.code"] = response.get("code", None)
                row_values[f"Response.{i}.probabilite"] = response.get("probabilite", None)
                row_values[f"Response.{i}.libelle"] = response.get("libelle", None)
        row_values["Response.IC"] = result.get("IC", None)

        # Add the dictionary of values to the list of predictions
        predictions.append(row_values)

    # Create a DataFrame from the list of predictions
    prediction_df = pd.DataFrame(predictions)

    # Add the prediction columns to the original DataFrame
    df = pd.concat([df, prediction_df], axis=1)

    return df


def main(data_file_path: str, dashboard_path:str, api_path:str) : #, date_to_log: str):
    # Define file system
    fs = get_filesystem()

    # Open Dataset
    data = (
        ds.dataset(
            f"{data_file_path}",
            partitioning=["date"],
            format="parquet",
            filesystem=fs,
        )
        .to_table()
        # .filter(
        #     (ds.field("date") == f"date={date_to_log}")
        # )
        .to_pandas()
    )

    # Harmonize dataset for the query
    table = format_query(data)
    results = query_batch_api(table, api_path, prob_min=0.0)
    table = add_prediction_columns(data, results)
    # Remove 'date=' prefix from the 'date' column to partition again
    table['date'] = table['date'].str.replace('date=', '')
    arrow_table = pa.Table.from_pandas(table)
    save_to_s3(arrow_table, "projet-ape", f"/{dashboard_path}/")


if __name__ == "__main__":
    data_file_path = str(sys.argv[1])
    dashboard_path = str(sys.argv[2])
    api_path = str(sys.argv[3])

    main(data_file_path, dashboard_path) #, date_to_log)
