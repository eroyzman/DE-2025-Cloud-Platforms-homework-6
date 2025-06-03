import boto3
import json
import os
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.embeddings import BedrockEmbeddings

# Initialize clients
textract = boto3.client('textract', region_name='us-east-1')
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

def extract_text(bucket, key):
    response = textract.start_document_text_detection(
        DocumentLocation={'S3Object': {'Bucket': bucket, 'Name': key}}
    )
    job_id = response['JobId']
    while True:
        result = textract.get_document_text_detection(JobId=job_id)
        if result['JobStatus'] == 'SUCCEEDED':
            break
    return result

def get_text_blocks(result, pdf_name):
    texts = []
    metadatas = []
    for block in result['Blocks']:
        if block['BlockType'] == 'LINE':
            texts.append(block['Text'])
            metadatas.append({"source": pdf_name, "page": block.get('Page', 1)})
    return texts, metadatas

def lambda_handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        if not key.startswith('company_data/'):  # Process only Company Data PDFs
            continue
        textract_result = extract_text(bucket, key)
        texts, metadatas = get_text_blocks(textract_result, key)
        bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", region_name="us-east-1")
        vector_db = OpenSearchVectorSearch(
            opensearch_url=os.environ['OPENSEARCH_ENDPOINT'],
            index_name="company-data-index",
            embedding=bedrock_embeddings,
            is_aoss=True
        )
        vector_db.add_texts(texts=texts, metadatas=metadatas)
    return {
        'statusCode': 200,
        'body': json.dumps('Processed PDF successfully')
    }