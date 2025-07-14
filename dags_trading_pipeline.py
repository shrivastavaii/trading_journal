from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'you',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='trading_journal_pipeline',
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False
) as dag:

    upload_webull = BashOperator(
        task_id='clean_and_upload_webull',
        bash_command='python3 /path/to/clean_data.py'
    )

    fetch_tsla_data = BashOperator(
        task_id='fetch_tsla_5min',
        bash_command='python3 /path/to/market_data.py TSLA 5min'
    )

    upload_webull >> fetch_tsla_data  # Run in order
