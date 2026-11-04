import argparse
import datetime
import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
import io
import re
import csv

import boto3
import pandas as pd
from dotenv import load_dotenv

def generate_date_conditions(start_date: datetime.date, end_date: datetime.date) -> str:
    """Generate the SQL WHERE clause for the date range."""
    conditions = []
    current_date = start_date
    while current_date <= end_date:
        conditions.append(
            f"(year = {current_date.year} AND month = {current_date.month} AND day = {current_date.day})"
        )
        current_date += datetime.timedelta(days=1)
    
    return " OR\n      ".join(conditions)

def fix_quoted_newlines(text):
    result = []
    in_quotes = False
    i = 0
    while i < len(text):
        char = text[i]
        if char == '"':
            in_quotes = not in_quotes
            result.append(char)
        elif in_quotes and char in ("\n", "\r", "\t"):
            result.append(" ")
        else:
            result.append(char)
        i += 1
    return "".join(result)

def clean_text_val(val):
    if pd.isna(val):
        return val
    val = str(val)
    val = re.sub(r"<[^>]+>", " ", val)
    val = re.sub(r"&[a-zA-Z]+;", " ", val)
    val = val.replace('""', '"')
    val = re.sub(r"[\r\n\t]+", " ", val)
    val = re.sub(r"\s+", " ", val)
    return val.strip()

def clean_and_save_data(filepath: Path) -> Path:
    print(f"\n--- Cleaning Scraped Data: {filepath} ---")
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    print(f"Raw size: {len(raw):,} chars")
    fixed = fix_quoted_newlines(raw)
    print("Fixed embedded newlines/tabs inside quoted text")

    df = pd.read_csv(
        io.StringIO(fixed),
        engine="python",
        on_bad_lines="skip"
    )

    print(f"Loaded rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    numerical_cols = {
        "display_size_inch", "screen_resolution", "display_refresh_hz", 
        "price_inr", "back_camera_mp_total", "front_camera_mp", 
        "cpu_cores", "antutu_score", "ram_gb", "storage_gb", 
        "battery_mah", "nfc", "5g_supported", "months_since_launch"
    }

    mode_exclude_cols = {"price_inr", "ram_gb", "storage_gb", "months_since_launch","launch_date"}

    def is_missing_string(val) -> bool:
        if pd.isna(val):
            return True
        s = str(val).strip().lower()
        return s in {"", "none", "null", "nan"}

    for col in df.columns:
        if col in numerical_cols:
            temp = pd.to_numeric(df[col], errors='coerce')
            if col not in mode_exclude_cols:
                non_null_vals = temp.dropna()
                if not non_null_vals.empty:
                    modes = non_null_vals.mode()
                    if not modes.empty:
                        col_mode = modes.iloc[0]
                        temp = temp.fillna(col_mode)
            # If the column became a float but all non-null values are whole numbers,
            # cast to 'Int64' (nullable integer) so it writes '8' instead of '8.0' to CSV.
            if pd.api.types.is_float_dtype(temp) and (temp.dropna() % 1 == 0).all():
                df[col] = temp.astype("Int64")
            else:
                df[col] = temp
        else:
            df[col] = df[col].apply(clean_text_val)
            df[col] = df[col].apply(lambda x: "U" if is_missing_string(x) else x)

    output_path = filepath.with_name(f"{filepath.stem}_clean.csv")
    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL
    )

    print(f"Saved cleaned data: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}\n")
    return output_path

def run_script(script_name: str) -> None:
    """Run a standalone python script as a subprocess."""
    print(f"\n=========================================")
    print(f"Running downstream script: {script_name}")
    print(f"=========================================")
    try:
        # sys.executable ensures we use the same Python interpreter/venv
        subprocess.run([sys.executable, script_name], check=True)
        print(f"Finished {script_name} successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Script {script_name} failed with return code {e.returncode}")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Automate fetching new devices from Athena and running LLM Scraper.")
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--limit", type=int, help="Optional limit for the number of devices to fetch and scrape.")
    parser.add_argument("--s3-filename", help="Optional custom filename for the final complete table in S3.")
    args = parser.parse_args()

    try:
        start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format.")
        sys.exit(1)

    if start_date > end_date:
        print("Error: start-date must be before or equal to end-date.")
        sys.exit(1)

    # Load environment variables
    load_dotenv()
    
    aws_access_key = os.environ.get("AWS_ACCESS_KEY")
    aws_secret_key = os.environ.get("AWS_SECRET_KEY")
    aws_region = os.environ.get("AWS_REGION")
    athena_s3_output = os.environ.get("ATHENA_S3_OUTPUT")
    athena_database = os.environ.get("ATHENA_DATABASE", "mobavenue_dsp")

    if not all([aws_access_key, aws_secret_key, aws_region, athena_s3_output]):
        print("Error: Missing AWS configuration in .env file.")
        print("Ensure AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, and ATHENA_S3_OUTPUT are set.")
        sys.exit(1)

    # 1. Generate Query
    print(f"Generating Athena query for dates between {start_date} and {end_date}...")
    date_conditions = generate_date_conditions(start_date, end_date)
    
    query = f"""
SELECT DISTINCT 
    LOWER(TRIM(device_model)) AS device_model,
    LOWER(TRIM(device_manufacturer)) AS device_manufacturer
FROM {athena_database}.rtb_bids
WHERE (
    {date_conditions}
)


EXCEPT

SELECT DISTINCT 
    LOWER(TRIM(device_model)) AS device_model,
    LOWER(TRIM(device_manufacturer)) AS device_manufacturer
FROM prod.device_features
    """.strip()
    
    if args.limit:
        query += f"\nLIMIT {args.limit};"
    else:
        query += ";"
    
    print("\n--- Query to Execute ---")
    print(query)
    print("------------------------\n")

    # 2. Execute Athena Query
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    athena_client = session.client('athena')
    
    print("Starting Athena query execution...")
    try:
        response = athena_client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': athena_database},
            ResultConfiguration={'OutputLocation': athena_s3_output}
        )
        query_execution_id = response['QueryExecutionId']
        print(f"Query Execution ID: {query_execution_id}")
    except Exception as e:
        print(f"Error starting Athena query: {e}")
        sys.exit(1)

    # 3. Poll for Completion
    print("Waiting for query to complete...", end="", flush=True)
    output_location = None
    while True:
        try:
            res = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
            state = res['QueryExecution']['Status']['State']
            
            if state == 'SUCCEEDED':
                print(f"\nQuery SUCCEEDED.")
                output_location = res['QueryExecution']['ResultConfiguration']['OutputLocation']
                break
            elif state in ['FAILED', 'CANCELLED']:
                reason = res['QueryExecution']['Status'].get('StateChangeReason', 'Unknown reason')
                print(f"\nQuery {state}. Reason: {reason}")
                sys.exit(1)
            else:
                print(".", end="", flush=True)
                time.sleep(3)
        except Exception as e:
            print(f"\nError polling query status: {e}")
            sys.exit(1)

    # 4. Download Results
    print(f"Downloading query results from {output_location}...")
    parsed_url = urllib.parse.urlparse(output_location)
    bucket = parsed_url.netloc
    key = parsed_url.path.lstrip('/')
    
    output_dir = Path("data/input")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_filename = f"new_devices_from_athena_{args.start_date}_to_{args.end_date}.csv"
    csv_file_path = output_dir / csv_filename
    
    s3_client = session.client('s3')
    try:
        s3_client.download_file(bucket, key, str(csv_file_path))
        print(f"Results downloaded to: {csv_file_path}")
    except Exception as e:
        print(f"Error downloading CSV from S3: {e}")
        sys.exit(1)
        
    # 5. Validate downloaded CSV
    try:
        df = pd.read_csv(csv_file_path)
        if len(df) == 0:
            print("No new devices found in this date range. Exiting.")
            sys.exit(0)
        print(f"Found {len(df)} new devices to process.")
    except Exception as e:
        print(f"Error reading downloaded CSV: {e}")
        sys.exit(1)

    # 6. Execute the LLM Scraper
    print("\nStarting the LLM Scraper...")
    scraper_cmd = [sys.executable, "app/main.py", "--input", str(csv_file_path)]
    print(f"Running command: {' '.join(scraper_cmd)}")
    
    try:
        # We use subprocess.run so the user can see the stdout/stderr stream directly
        result = subprocess.run(scraper_cmd)
        if result.returncode != 0:
            print(f"\nScraper exited with non-zero status code: {result.returncode}")
            sys.exit(result.returncode)
        else:
            print("\nScraper completed successfully!")
            
            # 7. Clean the output data
            output_csv_filename = f"{csv_file_path.stem}_output.csv"
            output_csv_path = Path("data/output") / output_csv_filename
            
            if not output_csv_path.exists():
                print(f"Warning: Expected output CSV not found at {output_csv_path}")
                print("Skipping cleaning and S3 upload.")
            else:
                clean_csv_path = clean_and_save_data(output_csv_path)
                
                # 8. Upload to S3
                print(f"\nUploading cleaned data to S3...")
                s3_bucket = "mobavenue-simplismart-aws-s3-apse-sg"
                s3_key = f"rtb/data/mis/device_feature/{clean_csv_path.name}"
                try:
                    s3_client.upload_file(str(clean_csv_path), s3_bucket, s3_key)
                    print(f"Successfully uploaded {clean_csv_path.name} to s3://{s3_bucket}/{s3_key}")
                except Exception as e:
                    print(f"Error uploading to S3: {e}")
                    sys.exit(1)
                    
                # 8.5 Prepare files for downstream combination and enrichment scripts
                file_a = Path("data/output/all_device_model_output.csv")
                file_b = Path("data/output/pending_device_output.csv")
                
                # Copy newly scraped and cleaned output to all_device_model_output.csv
                print(f"\nCopying {clean_csv_path.name} to {file_a.name} for downstream processing...")
                import shutil
                shutil.copy(str(clean_csv_path), str(file_a))
                
                # If pending_device_output.csv doesn't exist, create an empty CSV placeholder with matching headers
                if not file_b.exists():
                    print(f"Creating empty CSV placeholder for {file_b.name}...")
                    df_headers = pd.read_csv(clean_csv_path, nrows=0)
                    df_headers.to_csv(file_b, index=False)
                    
                # If unique_apps.csv doesn't exist, create a minimal placeholder
                unique_apps_file = Path("data/output/unique_apps.csv")
                if not unique_apps_file.exists():
                    print(f"Creating minimal CSV placeholder for {unique_apps_file.name}...")
                    pd.DataFrame(columns=["device_manufacturer", "device_model"]).to_csv(unique_apps_file, index=False)
                    
                # If device_specs.csv doesn't exist, create a minimal placeholder
                device_specs_file = Path("data/output/device_specs.csv")
                if not device_specs_file.exists():
                    print(f"Creating minimal CSV placeholder for {device_specs_file.name}...")
                    pd.DataFrame(columns=["device_manufacturer", "device_model", "launch_date", "cooling_system", "price_inr"]).to_csv(device_specs_file, index=False)
                    
                # 9. Run downstream orchestration scripts
                run_script("combine_outputs.py")
                run_script("enrich_from_unique_apps.py")
                run_script("enrich_from_device_specs.py")
                run_script("compact_enriched_output.py")
                run_script("split_output.py")

                # 10. Upload final complete table to S3 location
                final_complete_path = Path("data/output/final_complete.csv")
                if final_complete_path.exists():
                    print(f"\nUploading final complete table to S3...")
                    s3_dest_bucket = "mobavenue-simplismart-aws-s3-apse-sg"
                    
                    now = datetime.datetime.now()
                    date_folder = now.strftime("%Y%m%d")
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    s3_filename = args.s3_filename if args.s3_filename else f"{timestamp}.csv"
                    s3_dest_key_base = f"rtb/data/ml/device_features/{date_folder}/{s3_filename}"
                    
                    try:
                        # Upload to base folder
                        s3_client.upload_file(str(final_complete_path), s3_dest_bucket, s3_dest_key_base)
                        print(f"Successfully uploaded final complete table to s3://{s3_dest_bucket}/{s3_dest_key_base}")
                    except Exception as e:
                        print(f"Error uploading final complete table to S3: {e}")
                        sys.exit(1)
                else:
                    print(f"Warning: final_complete.csv not found at {final_complete_path}")
                    
            print("\nAutomation completed successfully!")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Error executing scraper: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
