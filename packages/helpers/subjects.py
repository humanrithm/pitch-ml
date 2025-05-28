import os
import json
from typing import Union
from connections import AWS

__version__ = '0.1.4'

# get all subject IDs
def list_subject_ids(aws_connection: AWS) -> list[str]:
    # get all subject IDs
    subject_query = """ 
        SELECT 
            subject_id
        FROM 
            info.subject_log
        WHERE subject_id IN (
            SELECT DISTINCT
                CAST(SPLIT_PART(study_id, '_', 1) AS INTEGER) 
            FROM 
                mocap.clean
        );
    """
    subject_ids = aws_connection.run_query(subject_query)

    return list(subject_ids['subject_id'])

# get all study IDs
def list_study_ids(
        aws_connection: AWS, 
        subject_id: str = None
) -> list[str]:
    if subject_id:
        study_ids = aws_connection.run_query(f"SELECT DISTINCT study_id FROM mocap.clean WHERE study_id LIKE '{subject_id}%';")
    else:
        study_ids = aws_connection.run_query("SELECT DISTINCT study_id FROM mocap.clean;")

    return list(study_ids['study_id'])