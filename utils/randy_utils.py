import pandas as pd
# pip install google-cloud-secret-manager
from google.cloud import bigquery, secretmanager
from google.oauth2 import service_account
from google.api_core.exceptions import NotFound



def upload_df_to_bq(
    df: pd.DataFrame,
    billing_project_id: str,
    dataset_project_id: str,
    dataset_id: str,
    table_id: str,
    service_account_key_info=None,
    service_account_key_path: str = None,
    write_disposition="WRITE_APPEND"
) -> str:
    """
    Uploads a pandas DataFrame to a BigQuery table using a specific service account.

    Args:
        df (pd.DataFrame): Data to upload.
        billing_project_id (str): Project to bill for the load job.
        dataset_project_id (str): Project where the target dataset resides.
        dataset_id (str): Target dataset.
        table_id (str): Target table.
        service_account_key_info (dict): Dict-form key (from Secret Manager, for example).
        service_account_key_path (str): File path to key (if not using key_info).
        write_disposition (str): "WRITE_APPEND", "WRITE_TRUNCATE", or "WRITE_EMPTY".
    """
    df = df.astype(str)

    # Load credentials
    if service_account_key_path is None:
        if service_account_key_info is None:
            raise ValueError("Must provide service_account_key_path or service_account_key_info.")
        credentials = service_account.Credentials.from_service_account_info(service_account_key_info)
    else:
        credentials = service_account.Credentials.from_service_account_file(service_account_key_path)

    # Initialize BQ client with billing project
    client = bigquery.Client(project=billing_project_id, credentials=credentials)

    table_ref = f"{dataset_project_id}.{dataset_id}.{table_id}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        autodetect=True,
    )

    try:
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()
        return f"Uploaded {job.output_rows} rows to {table_ref}"
    except Exception as e:
        raise RuntimeError(f"Failed to upload to BigQuery: {e}")

def get_secret_from_secret_manager(project_id: str, secret_id: str, version_id: str = "latest") -> str:
    """
    Retrieve a secret value from Google Secret Manager.

    Args:
        project_id (str): GCP project ID that owns the secret.
        secret_id (str): The ID of the secret.
        version_id (str): The version of the secret to access (default is 'latest').

    Returns:
        str: The decoded secret payload as a string.

    Raises:
        NotFound: If the secret or version doesn't exist.
        Exception: For other issues accessing the secret.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    try:
        response = client.access_secret_version(request={"name": name})
        secret_value = response.payload.data.decode("UTF-8")
        return secret_value
    except NotFound as e:
        raise ValueError(f"Secret not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to access secret: {e}")
