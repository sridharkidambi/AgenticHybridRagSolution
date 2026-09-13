from langchain_community.retrievers import PineconeHybridSearchRetriever
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
import boto3
from botocore.exceptions import ClientError
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

index_name = "hybrid-search-langchain-pinecone"
load_dotenv(override=True)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f"Index '{index_name}' already exists.")

index = pc.Index(index_name)
print(f"Index '{index_name}' is ready.")

embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
print(embeddings)

BM25_PARAMS_PATH = "bm25_encoder.json"
if os.path.exists(BM25_PARAMS_PATH):
    bm25_encoder = BM25Encoder().load(BM25_PARAMS_PATH)
    print(f"Loaded BM25 encoder from '{BM25_PARAMS_PATH}'.")
else:
    bm25_encoder = BM25Encoder().default()
    print("Initialized default BM25 encoder.")

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_PDF_KEY = os.getenv("S3_PDF_KEY")

s3_client = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION", "eu-north-1"),
)


def s3_object_exists(bucket_name, key):
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("404", "NoSuchKey"):
            return False
        raise


def _rename_s3_key_with_indexed_suffix(bucket_name, key):
    if key.lower().endswith(".pdf"):
        new_key = f"{key[:-4]}_indexed.pdf"
    else:
        new_key = f"{key}_indexed"

    s3_client.copy_object(
        Bucket=bucket_name,
        CopySource={"Bucket": bucket_name, "Key": key},
        Key=new_key,
    )
    s3_client.delete_object(Bucket=bucket_name, Key=key)
    print(f"Renamed S3 object '{key}' to '{new_key}' in bucket '{bucket_name}'.")
    return new_key


if s3_object_exists(S3_BUCKET_NAME, S3_PDF_KEY):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        s3_client.download_fileobj(S3_BUCKET_NAME, S3_PDF_KEY, tmp_file)
        tmp_file_path = tmp_file.name

    print(f"Downloaded PDF from S3 bucket '{S3_BUCKET_NAME}' key '{S3_PDF_KEY}' to '{tmp_file_path}'")

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} page(s) from PDF.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split PDF into {len(chunks)} chunks.")

    os.remove(tmp_file_path)

    chunk_texts = [chunk.page_content for chunk in chunks]
    bm25_encoder.fit(chunk_texts)
    bm25_encoder.dump(BM25_PARAMS_PATH)

    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25_encoder,
        index=index,
    )

    try:
        retriever.add_texts(chunk_texts)
        print(f"Added {len(chunk_texts)} chunks to Pinecone index '{index_name}'.")
    except Exception as e:
        print(f"Failed to add texts using existing retriever/index reference: {e}")
        print("Refreshing index reference from Pinecone DB...")
        index = pc.Index(index_name)
        retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings,
            sparse_encoder=bm25_encoder,
            index=index,
        )
        retriever.add_texts(chunk_texts)
        print(f"Added {len(chunk_texts)} chunks to Pinecone index '{index_name}' after refreshing index reference.")

    _rename_s3_key_with_indexed_suffix(S3_BUCKET_NAME, S3_PDF_KEY)
else:
    print(f"S3 file '{S3_PDF_KEY}' not found in bucket '{S3_BUCKET_NAME}'. It has already been indexed.")
    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25_encoder,
        index=index,
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""
)

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), temperature=0)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "find my joining date"
response = rag_chain.invoke(query)
print("\n--- LLM Response ---")
print(response)

