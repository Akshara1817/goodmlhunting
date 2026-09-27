"""
SageMaker Training Launcher (launch_sagemaker.py)
Executes from your local machine to run training on AWS without wasting credits.
"""

import boto3
import sagemaker
from sagemaker.estimator import Estimator

# 1. AWS Session setup
session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()
prefix = 'amazon-ml-challenge-2026'

print(f"Connected to AWS S3 Bucket: {bucket}")

# 2. Upload training directory to S3
print("Uploading train dataset to S3...")
train_s3_uri = session.upload_data(
    path='student_resource/dataset/train',
    bucket=bucket,
    key_prefix=f'{prefix}/train'
)
print(f"Train data ready at: {train_s3_uri}")

# 3. Define the Estimator
# Uses an optimized compute instance (ml.c5.4xlarge, 16 vCPUs)
estimator = Estimator(
    entry_point='train.py',
    source_dir='goodmlhunting/code/src',
    role=role,
    instance_count=1,
    instance_type='ml.c5.4xlarge',
    framework_version='1.0',
    py_version='py310',
    hyperparameters={
        'neg-ratio': 5
    },
    base_job_name='entity-resolution-lgbm',
    use_spot_instances=True,  # Saves up to 70% of AWS credits
    max_wait=3600,
    max_run=3600
)

# 4. Trigger cloud training
print("Starting SageMaker Training Job...")
estimator.fit({'train': train_s3_uri})

print(f"Training completed! Model artifact stored in: {estimator.model_data}")