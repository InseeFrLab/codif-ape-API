import os
import sys

import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
from tqdm import tqdm


def transform_json_to_dataframe(json_dir: str):

    transformed_data = []

    for filename in tqdm(os.listdir(json_dir)):
        with open(os.path.join(json_dir, filename), 'r') as file:
            data = json.load(file)

        annotation_date = data['task']['updated_at']
        # Get data variables
        liasse_numero = data['task']['data']['liasse_numero']
        date_modification = data['task']['data']['date_modification']
        mode_calcul_ape = data['task']['data']['mode_calcul_ape']
        evenement_type = data['task']['data']['evenement_type']
        liasse_type = data['task']['data']['liasse_type']
        activ_surf_et = str(data['task']['data']['activ_surf_et'])
        activ_nat_et = data['task']['data']['activ_nat_et']
        cj = data['task']['data']['cj']

        # Extract text description amongst three columns according to the order of preference.
        activ_pr_lib_et = data['task']['data'].get('activ_pr_lib_et')
        activ_ex_lib_et = data['task']['data'].get('activ_ex_lib_et')
        activ_pr_lib = data['task']['data'].get('activ_pr_lib')

        if activ_pr_lib_et:
            libelle = activ_pr_lib_et
        elif activ_ex_lib_et:
            libelle = activ_ex_lib_et
        else:
            libelle = activ_pr_lib

        # Number of skips
        skips = int(data['was_cancelled'])
        # Get annotated data without skips and adjust from UI's bugs
        if len(data['result']) > 0:
            apet_manual = ""
            commentary = ""
            # Retrieve comment
            if 'text' in data['result'][0]['value']:        
                commentary = data['result'][0]['value']['text'][0]
            # Retrieve taxonomy result
            if 'taxonomy' in data['result'][0]['value']:
                taxonomy_values = data['result'][0]['value']['taxonomy'][0][-1]
                apet_manual = taxonomy_values.replace('.', '') # delete . in apet_manual
            #Check if apet is in comment and fill empty apet (due to LS bug)
            if commentary[:4].isdigit() and commentary[4].isalpha():
                print("Potential LS bug --> NAF code in comment: "+ commentary)
                apet_manual = commentary

        # Créer un dictionnaire pour les données transformées
        transformed_row = {
            'liasse_numero' : liasse_numero,
            'libelle': libelle,
            'evenement_type' : evenement_type,
            'liasse_type' : liasse_type,
            'activ_surf_et' : activ_surf_et,
            'activ_nat_et' : activ_nat_et,
            'cj' : cj,
            'date_modification' : date_modification,
            'annotation_date' : annotation_date,
            'mode_calcul_ape' : mode_calcul_ape,
            'apet_manual': apet_manual,
            'commentary' : commentary,
            'skips' : skips,
        }
        
        # Append in list 
        transformed_data.append(transformed_row)

    # Convert to Dataframe
    results = pd.DataFrame(transformed_data)

    # Keep only unskipped and classifiable annotations
    results = results[results['skips'] == 0]
    results = results[results['apet_manual'] != "XXXXX"]
    print(len(results))

    return results


def save_to_s3(table: pa.Table, bucket: str, path: str):
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"},
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    # To partition data
    pq.write_to_dataset(
        table,
        root_path=f"s3://{bucket}/{path}/",
        partition_cols=["date"],
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        filesystem=fs,
    )
    pq.write_table(table,f"s3://{bucket}/{path}/test_data_NAF2008.parquet",filesystem=fs)


def main(annotation_results_path: str, annotation_preprocessed_path: str):
    # Upload log file from API pod and filter by date
    data = transform_json_to_dataframe(annotation_results_path)

    # Add date column for partitionning
    data["date"] = pd.to_datetime(data["date_modification"]).dt.strftime("%Y-%m-%d")

    data = (
        data[
            [
                "liasse_numero",
                "libelle",
                "evenement_type",
                "liasse_type",
                "activ_surf_et",
                "activ_nat_et",
                "cj",
                "date",
                "mode_calcul_ape",
                "apet_manual",
           
            ]
        ]
        .rename(
            columns={
                "libelle": "text_description",
                "liasse_type": "type_",
                "activ_nat_et": "nature",
                "activ_surf_et": "surface",
                "evenement_type": "event",
            }
        )
        .reset_index(drop=True)
    )

    # data['surface'] = data['surface'].astype(bytes)
    # Translate pd.DataFrame into pa.Table
    arrow_table = pa.Table.from_pandas(data)

    # Save logs in a partionned parquet file in s3
    save_to_s3(arrow_table, "projet-ape", f"/{annotation_preprocessed_path}/")


if __name__ == "__main__":
    annotation_results_path = str(sys.argv[1])
    annotation_preprocessed_path = str(sys.argv[2])

    main(annotation_results_path, annotation_preprocessed_path)
