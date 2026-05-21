import argparse
import datetime
import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

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

def main():
    parser = argparse.ArgumentParser(description="Automate fetching new devices from Athena and running LLM Scraper.")
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD format")
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
SELECT DISTINCT LOWER(TRIM(device_model)) as device_model , LOWER(TRIM(device_manufacturer)) as device_manufacturer
FROM {athena_database}.rtb_bids
WHERE (
      {date_conditions}
)
EXCEPT
SELECT DISTINCT LOWER(TRIM(device_model)) as device_model , LOWER(TRIM(device_manufacturer)) as device_manufacturer
FROM imp_tables.device_specs_iceberg;
    """.strip()
    
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
    # print("\nStarting the LLM Scraper...")
    # scraper_cmd = [sys.executable, "app/main.py", "--input", str(csv_file_path)]
    # print(f"Running command: {' '.join(scraper_cmd)}")
    
    # try:
    #     # We use subprocess.run so the user can see the stdout/stderr stream directly
    #     result = subprocess.run(scraper_cmd)
    #     if result.returncode != 0:
    #         print(f"\nScraper exited with non-zero status code: {result.returncode}")
    #         sys.exit(result.returncode)
    #     else:
    #         print("\nAutomation completed successfully!")
    # except KeyboardInterrupt:
    #     print("\nProcess interrupted by user.")
    #     sys.exit(130)
    # except Exception as e:
    #     print(f"Error executing scraper: {e}")
    #     sys.exit(1)

if __name__ == "__main__":
    main()
