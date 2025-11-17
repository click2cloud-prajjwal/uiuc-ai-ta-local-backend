import os
import re
import json
import io
import time
import uuid
import shutil
import logging
import asyncio
import inspect
import traceback
import mimetypes
import subprocess
from typing import Any, Callable, Dict, List, Optional, Union, cast
from pathlib import Path
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import openai
import sentry_sdk
# from posthog import Posthog
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

# Updated LangChain imports
from langchain_community.document_loaders import (
      Docx2txtLoader,
      GitLoader,
      PythonLoader,
      TextLoader,
      UnstructuredExcelLoader,
      UnstructuredPowerPointLoader,
      CSVLoader,
)

from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant

import fitz
import pdfplumber
import pytesseract
from PIL import Image
from git.repo import Repo
from bs4 import BeautifulSoup
from pydub import AudioSegment

try:
    from rabbitmq.rmsql import SQLAlchemyIngestDB
    from rabbitmq.embeddings import OpenAIAPIProcessor
except ModuleNotFoundError:
    # When running as worker outside Flask app, import from local path
    from rmsql import SQLAlchemyIngestDB
    from embeddings import OpenAIAPIProcessor

load_dotenv(override=False)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Ingest:
    """
    Class for ingesting documents into the vector database.
    """

    def __init__(self):
        # Azure OpenAI credentials (standardized)
        self.azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.azure_openai_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.embedding_model = os.getenv('EMBEDDING_MODEL')
        
        # Qdrant credentials
        self.qdrant_url = os.getenv('QDRANT_URL','http://qdrant:6333')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY')
        self.qdrant_collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'uiuc-chatbot')

        whisper_endpoint = os.getenv('AZURE_WHISPER_ENDPOINT')
        whisper_key = os.getenv('AZURE_WHISPER_KEY')
        
        if whisper_endpoint and whisper_key:
            from openai import AzureOpenAI
            self.whisper_client = AzureOpenAI(
                api_key=whisper_key,
                azure_endpoint=whisper_endpoint
            )
            self.whisper_deployment = os.getenv('AZURE_WHISPER_DEPLOYMENT', 'whisper')
            logging.info("✅ Whisper client initialized successfully")
        else:
            self.whisper_client = None
            self.whisper_deployment = None
            logging.warning("⚠️ Whisper credentials not found. Audio/video ingestion will be skipped.")
        
                # === Cross-platform Tesseract setup ===
        try:
            if os.name == "nt":  # Windows
                # Typical Windows installation path
                tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                if os.path.exists(tesseract_path):
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                    logging.info(f" Windows: Tesseract found at {tesseract_path}")
                else:
                    logging.warning("Windows: Tesseract not found at default path. OCR features may not work.")
            else:
                # Linux / macOS / Docker (Debian slim)
                result = shutil.which("tesseract")

                if result:
                    pytesseract.pytesseract.tesseract_cmd = result
                    logging.info(f" Linux: Tesseract found at {result}")
                else:
                    logging.warning("Linux: Tesseract not found in PATH. Attempting to install...")

                    # Attempt to install tesseract automatically
                    try:
                        subprocess.run(["apt-get", "update"], check=False)
                        subprocess.run(["apt-get", "install", "-y", "tesseract-ocr"], check=True)

                        # Verify again
                        result = shutil.which("tesseract")
                        if result:
                            pytesseract.pytesseract.tesseract_cmd = result
                            logging.info(f" Successfully installed Tesseract at {result}")
                        else:
                            logging.error(" Tesseract installation completed but binary not found in PATH.")
                    except Exception as install_err:
                        logging.error(f" Failed to install Tesseract automatically: {install_err}")

        except Exception as e:
            logging.error(f" Error while configuring Tesseract OCR: {e}")


        # Runtime objects
        # self.posthog = None
        self.qdrant_client = None
        self.vectorstore = None
        self.blob_client = None
        self.sql_session = None

        # Qdrant ingestion tuning
        self.qdrant_upsert_batch_size = int(os.getenv('QDRANT_UPSERT_BATCH_SIZE', '100'))
        self.qdrant_indexing_threshold_ingest = int(os.getenv('QDRANT_INDEXING_THRESHOLD_INGEST', '100000000'))
        self.qdrant_indexing_threshold_online = int(os.getenv('QDRANT_INDEXING_THRESHOLD_ONLINE', '1000'))

    def get_tesseract_lang_from_docgroups(self, doc_groups):
        """
        Map doc_groups to appropriate Tesseract OCR language codes.
        Returns a single tesseract language code (e.g. 'hin', 'mar', 'eng').
        """
        # Normalize to list
        if not doc_groups:
            return "eng"

        if isinstance(doc_groups, str):
            doc_list = [doc_groups]
        else:
            doc_list = list(doc_groups) if hasattr(doc_groups, '__iter__') else [str(doc_groups)]

        # pick the first meaningful group
        doc_group = str(doc_list[0]).lower().strip()

        lang_map = {
            "hindi": "hin",
            "marathi": "mar",
            "punjabi": "pan",
            "gujarati": "guj",
            "telugu": "tel",
            "tamil": "tam"
        }

        for key, val in lang_map.items():
            if key in doc_group:
                return val

        # fallback to English
        return "eng"


    def initialize_resources(self):
        """Initialize Qdrant client and vectorstore with Azure OpenAI embeddings"""
        
        # Initialize Qdrant client and create collection if necessary
        # Detect if using local Qdrant
        is_local_qdrant = "localhost" in self.qdrant_url or "127.0.0.1" in self.qdrant_url
        
        # Initialize Qdrant client
        if is_local_qdrant:
            # Local Qdrant (no API key needed)
            logging.info("?? Using LOCAL Qdrant instance")
            self.qdrant_client = QdrantClient(
                url=self.qdrant_url,
                api_key=None,
                timeout=20
            )
        elif self.qdrant_url:  # ? REMOVED api_key requirement
            # Cloud/Docker Qdrant (API key optional)
            logging.info("?? Using Qdrant instance at: " + self.qdrant_url)
            self.qdrant_client = QdrantClient(
                url=self.qdrant_url, 
                api_key=self.qdrant_api_key if self.qdrant_api_key else None,
                timeout=20
            )
        else:
            logging.error("? QDRANT URL NOT FOUND!")
            
        try:
            collection_info = self.qdrant_client.get_collection(self.qdrant_collection_name)
            logging.info(f"Collection {self.qdrant_collection_name} exists with {collection_info.points_count} points")
        except Exception as e:
            logging.info(f"Creating collection {self.qdrant_collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.qdrant_collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)  # text-embedding-3-small = 1536
            )
            
            # Create indexes for filtering
            self.qdrant_client.create_payload_index(
                collection_name=self.qdrant_collection_name,
                field_name="conversation_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.qdrant_client.create_payload_index(
                collection_name=self.qdrant_collection_name,
                field_name="course_name",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logging.info("Created payload indexes for conversation_id and course_name")


        
        # Initialize Azure OpenAI embeddings
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            model=os.environ["EMBEDDING_MODEL"],
            deployment=os.environ["EMBEDDING_MODEL"]
        )

        self.vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name=self.qdrant_collection_name,
            embeddings=embeddings
        )
        logging.info(" Vectorstore initialized with Azure OpenAI embeddings.")


        # ===== AZURE BLOB STORAGE =====
        try:
            account_name = os.getenv("AZURE_SA_NAME")
            account_key = os.getenv("AZURE_SA_ACCESSKEY")
            connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
            self.blob_container = os.getenv("AZURE_BLOB_CONTAINER", "uiuc-chatbot")

            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            self.blob_client = blob_service_client.get_container_client(self.blob_container)

            # Ensure container exists
            if not self.blob_client.exists():
                self.blob_client.create_container()

            logging.info(f" Connected to Azure Blob Storage (container={self.blob_container})")
        except Exception as e:
            logging.error(f" Failed to connect to Azure Blob Storage: {e}")


        def get_object(*args, **kwargs):
            key = kwargs.get("Key") or args[1]
            blob_client = self.blob_client.get_blob_client(key)
            blob_data = blob_client.download_blob().readall()
            return {"Body": io.BytesIO(blob_data)}

        def download_fileobj(*args, **kwargs):
            key = kwargs.get("Key") or args[1]
            fileobj = kwargs.get("Fileobj") or args[2]
            blob_client = self.blob_client.get_blob_client(key)
            stream = blob_client.download_blob()
            fileobj.write(stream.readall())

        def upload_fileobj(*args, **kwargs):
            fileobj = args[0]
            key = kwargs.get("Key") or args[2]
            blob_client = self.blob_client.get_blob_client(key)
            blob_client.upload_blob(fileobj, overwrite=True)

        def delete_object(*args, **kwargs):
            key = kwargs.get("Key") or args[1]
            blob_client = self.blob_client.get_blob_client(key)
            blob_client.delete_blob()

        self.blob_client.get_object = get_object
        self.blob_client.download_fileobj = download_fileobj
        self.blob_client.upload_fileobj = upload_fileobj
        self.blob_client.delete_object = delete_object

        logging.info(" Added blob-compatible methods to Azure Blob client for legacy ingestion.")

        self.sql_session = SQLAlchemyIngestDB()



    def main_ingest(self, job_id: str, **inputs: Dict[str | List[str], Any]):
        """
        Main ingest function.
        """
        try:
            self.initialize_resources()

            course_name: List[str] | str = inputs.get('course_name', '')
            blob_path: List[str] | str = inputs.get('blob_path', '')
            url: List[str] | str | None = inputs.get('url', None)
            base_url: List[str] | str | None = inputs.get('base_url', None)
            readable_filename: List[str] | str = inputs.get('readable_filename', '')
            force_embeddings: bool = inputs.get('force_embeddings', False)  # if content is duplicated, still rescan

            content: str | List[str] | None = inputs.get('content', None)  # defined if ingest type is webtext
            # Normalize document groups to a consistent list format
            doc_groups = inputs.get('groups') or inputs.get('doc_groups')

            # Ensure doc_groups is always a list
            if isinstance(doc_groups, str) and doc_groups.strip():
                doc_groups = [doc_groups]
            elif not doc_groups:
                doc_groups = []

            print(
                f"In top of /ingest route. course: {course_name}, blobpaths: {blob_path}, readable_filename: {readable_filename}, base_url: {base_url}, url: {url}, content: {content}, doc_groups: {doc_groups}"
            )
            success_fail_dict = self.run_ingest(course_name, blob_path, base_url, url, readable_filename, content,
                                                doc_groups, force_embeddings)
            for retry_num in range(1, 3):
                if isinstance(success_fail_dict, str):  # TODO: What does this indicate?
                    success_fail_dict = self.run_ingest(course_name, blob_path, base_url, url, readable_filename, content,
                                                        doc_groups,force_embeddings)
                    time.sleep(13 * retry_num)  # max is 65
                elif success_fail_dict['failure_ingest']:
                    logging.error(f"Ingest failure -- Retry attempt {retry_num}. File: {success_fail_dict}")
                    success_fail_dict = self.run_ingest(course_name, blob_path, base_url, url, readable_filename, content,
                                                        doc_groups,force_embeddings)
                    time.sleep(13 * retry_num)  # max is 65
                else:
                    break
            if success_fail_dict['failure_ingest']:
                logging.error(f"INGEST FAILURE -- About to send to database. success_fail_dict: {success_fail_dict}")
                self.sql_session.insert_failed_document({
                    "blob_path": str(blob_path),
                    "readable_filename": readable_filename,
                    "course_name": course_name,
                    "url": url,
                    "base_url": base_url,
                    "doc_groups": doc_groups,
                    "error": str(success_fail_dict),
                })

            # Remove from documents in progress
            self.sql_session.delete_document_in_progress(job_id)

            sentry_sdk.flush(timeout=20)
            return json.dumps(success_fail_dict)

        except Exception as e:
            logging.error("Error in main_ingest: ", e)
            sentry_sdk.capture_exception(e)
            success_fail_dict = {"failure_ingest": {'error': str(e)}}
            return json.dumps(success_fail_dict)

    def run_ingest(self, course_name, blob_path, base_url, url, readable_filename, content, document_groups,
                   force_embeddings=False):
        """Routes ingest jobs based on the input data -> webscrape, url, readable_filename"""
        if content:
            return self.ingest_single_web_text(course_name, base_url, url, content, readable_filename,
                                               groups=document_groups, force_embeddings=force_embeddings)
        elif readable_filename == '':
            return self.bulk_ingest(course_name, blob_path, base_url=base_url, url=url,
                                    groups=document_groups, force_embeddings=force_embeddings)
        else:
            return self.bulk_ingest(course_name, blob_path, base_url=base_url, url=url,
                                    groups=document_groups, readable_filename=readable_filename, force_embeddings=force_embeddings)

    def bulk_ingest(self, course_name: str, blob_path: Union[str, List[str]],
                force_embeddings: bool, **kwargs) -> Dict[str, None | str | Dict[str, str]]:
        """Bulk ingest from blob paths, direct URLs, OR uploaded files into the vectorstore and database."""
        print(f"Top of bulk_ingest: ", kwargs)
        # ✅ Normalize doc_groups early so it propagates correctly
        doc_groups = kwargs.get('groups') or kwargs.get('doc_groups') or []
        if isinstance(doc_groups, str) and doc_groups.strip():
            doc_groups = [doc_groups]
        kwargs['doc_groups'] = doc_groups
        kwargs['groups'] = doc_groups  # keep backward compatible


        def _ingest_single(ingest_method: Callable, blob_path: str, course_name: str, force_embeddings: bool, *args, **kwargs):
            """Handle running an arbitrary ingest function for an individual file."""
            ret = ingest_method(blob_path, course_name, force_embeddings, *args, **kwargs)
            if ret == "Success":
                success_status['success_ingest'].append(str(blob_path))
            else:
                success_status['failure_ingest'].append({
                    'blob_path': str(blob_path),
                    'error': str(ret)
                })


        file_ingest_methods = {
            '.html': self._ingest_html,
            '.py': self._ingest_single_py,
            '.pdf': self._ingest_single_pdf,
            '.txt': self._ingest_single_txt,
            '.md': self._ingest_single_txt,
            '.srt': self._ingest_single_srt,
            '.vtt': self._ingest_single_vtt,
            '.docx': self._ingest_single_docx,
            '.ppt': self._ingest_single_ppt,
            '.pptx': self._ingest_single_ppt,
            '.xlsx': self._ingest_single_excel,
            '.xls': self._ingest_single_excel,
            '.xlsm': self._ingest_single_excel,
            '.xlsb': self._ingest_single_excel,
            '.xltx': self._ingest_single_excel,
            '.xltm': self._ingest_single_excel,
            '.xlt': self._ingest_single_excel,
            '.xml': self._ingest_single_excel,
            '.xlam': self._ingest_single_excel,
            '.xla': self._ingest_single_excel,
            '.xlw': self._ingest_single_excel,
            '.xlr': self._ingest_single_excel,
            '.csv': self._ingest_single_csv,
            '.png': self._ingest_single_image,
            '.jpg': self._ingest_single_image,
            '.gitignore': self._ingest_single_txt,
        }

        mimetype_ingest_methods = {
            'video': self._ingest_single_video,
            'audio': self._ingest_single_video,
            'text': self._ingest_single_txt,
            'image': self._ingest_single_image,
        }

        success_status = {
            "success_ingest": [],
            "failure_ingest": []
        }

        
        try:
            if isinstance(blob_path, str):
                blob_path = [blob_path]

            # ====== NEW: HANDLE FILE UPLOAD â†’ MinIO ======
            uploaded_file_path = kwargs.get('uploaded_file_path', '')
            
            if uploaded_file_path and (not blob_path or blob_path[0] == ''):
                # File was uploaded via form-data, need to save to MinIO first
                logging.info(f"File upload detected: {uploaded_file_path}")
                
                try:
                    # Generate blob key with UUID
                    file_extension = Path(uploaded_file_path).suffix
                    blob_key = f"{course_name}/{uuid.uuid4()}{file_extension}"
                    
                    logging.info(f"Uploading to Azure Blob Storage: {blob_key}")
                    
                    # Upload to MinIO/blob
                    with open(uploaded_file_path, 'rb') as file_data:
                        blob_client = self.blob_client.get_blob_client(blob_key)
                        blob_client.upload_blob(file_data, overwrite=True)

                    
                    logging.info(f" Uploaded to Azure Blob: blob://{self.blob_container}/{blob_key}")
                    
                    # Update kwargs with blob_path for downstream processing
                    kwargs['blob_path'] = blob_key
                    
                    # Now set blob_path to process this file through existing blob flow
                    blob_path = [blob_key]
                    
                    # Clean up temp file
                    try:
                        os.unlink(uploaded_file_path)
                        logging.info(f"Cleaned up temp file: {uploaded_file_path}")
                    except Exception as cleanup_error:
                        logging.warning(f"Could not delete temp file: {cleanup_error}")
                    
                    # Continue to blob-based ingestion below with the uploaded file
                    
                except Exception as e:
                    err_msg = f"Failed to upload file to MinIO: {str(e)}"
                    logging.error(f" {err_msg}")
                    logging.error(traceback.format_exc())
                    success_status['failure_ingest'] = {'file': uploaded_file_path, 'error': err_msg}
                    
                    # Try to clean up temp file even on error
                    try:
                        os.unlink(uploaded_file_path)
                    except:
                        pass
                    
                    return success_status
            
            # ====== EXISTING: URL-BASED INGESTION ======
            url = kwargs.get('url', '')
            
            if url and (not blob_path or blob_path[0] == ''):
                # Direct URL ingestion (no blob)
                logging.info(f" URL-based ingestion detected: {url}")
                
                try:
                    import requests
                    
                    # Download from URL
                    logging.info(f" Downloading from URL: {url}")
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    # Detect file extension from URL
                    from urllib.parse import urlparse
                    url_path = urlparse(url).path
                    file_extension = Path(url_path).suffix or '.txt'
                    
                    logging.info(f"   Detected file extension: {file_extension}")
                    
                    # Save to temporary file
                    with NamedTemporaryFile(delete=False, suffix=file_extension) as tmpfile:
                        tmpfile.write(response.content)
                        tmpfile.flush()
                        temp_path = tmpfile.name
                    
                    logging.info(f" Downloaded to temp file: {temp_path}")
                    
                    # Determine ingest method
                    if file_extension in file_ingest_methods:
                        ingest_method = file_ingest_methods[file_extension]
                    else:
                        # Fallback to text
                        ingest_method = self._ingest_single_txt
                    
                    # Call the ingest method with temp file
                    logging.info(f" Processing with method: {ingest_method.__name__}")
                    ret = self._ingest_from_local_file(temp_path, course_name, **kwargs)
                    
                    # Cleanup
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                    
                    if ret == "Success":
                        success_status['success_ingest'] = url
                        logging.info(f" Successfully ingested from URL: {url}")
                    else:
                        success_status['failure_ingest'] = {'url': url, 'error': str(ret)}
                        logging.error(f" Failed to ingest from URL: {url}, error: {ret}")
                    
                    return success_status
                    
                except Exception as e:
                    err_msg = f"Failed to ingest from URL {url}: {str(e)}"
                    logging.error(f" {err_msg}")
                    logging.error(traceback.format_exc())
                    success_status['failure_ingest'] = {'url': url, 'error': err_msg}
                    return success_status
            
            # ====== EXISTING: INGESTION ======
            for blob_path in blob_path:
                logging.info(f" Processing from Blob: {blob_path}")
                
                file_extension = Path(blob_path).suffix
                
                #  CHANGED: Use get_object instead of download_fileobj
                with NamedTemporaryFile(suffix=file_extension, delete=False) as tmpfile:
                    response = self.blob_client.get_object(Bucket=self.blob_container, Key=blob_path)
                    tmpfile.write(response['Body'].read())
                    tmpfile.flush()
                    temp_file_path = tmpfile.name
                
                mime_type = str(mimetypes.guess_type(temp_file_path, strict=False)[0])
                mime_category = mime_type.split('/')[0] if '/' in mime_type else mime_type
                
                # Clean up temp file after determining mime type
                os.remove(temp_file_path)

                # Ingest with specialized functions when possible, fallback to mimetype.
                if file_extension in file_ingest_methods:
                    ingest_method = file_ingest_methods[file_extension]
                    _ingest_single(ingest_method,blob_path,course_name,force_embeddings,**{**kwargs, "doc_groups": kwargs.get("groups") or kwargs.get("doc_groups", [])})
                elif mime_category in mimetype_ingest_methods:
                    ingest_method = mimetype_ingest_methods[mime_category]
                    _ingest_single(ingest_method,blob_path,course_name,force_embeddings,**{**kwargs, "doc_groups": kwargs.get("groups") or kwargs.get("doc_groups", [])})
                else:
                    # No supported ingest... Fallback to attempting utf-8 decoding, otherwise fail.
                    try:
                        self._ingest_single_txt(blob_path, course_name, force_embeddings, **kwargs)
                        success_status['success_ingest'] = blob_path
                    except Exception as e:
                        err_msg = f"No ingest method for filetype: {file_extension} (with generic type {mime_type}), for file: {blob_path}"
                        success_status['failure_ingest'] = {
                            'blob_path': blob_path,
                            'error': err_msg
                        }
                        # if self.posthog:
                        #     self.posthog.capture('distinct_id_of_the_user', event='ingest_failure',
                        #         properties={
                        #             'course_name': course_name,
                        #             'blob_path': blob_path,
                        #             'kwargs': kwargs,
                        #             'error': err_msg
                                # })
            return success_status
            
        except Exception as e:
            err = f" Error in /ingest: `{inspect.currentframe().f_code.co_name}`: {e}\nTraceback:\n", traceback.format_exc()
            success_status['failure_ingest'] = {
                'blob_path': blob_path if 'blob_path' in locals() else 'unknown',
                'error': f"MAJOR ERROR DURING INGEST: {err}"
            }
            # if self.posthog:
            #     self.posthog.capture('distinct_id_of_the_user', event='ingest_failure',
            #                         properties={
            #                             'course_name': course_name,
            #                             'blob_path': blob_path,
            #                             'kwargs': kwargs,
            #                             'error': err
            #                         })
            sentry_sdk.capture_exception(e)
            return success_status


    def _ingest_from_local_file(self, file_path: str, course_name: str, **kwargs) -> str:
        """
        Ingest from a local file path (for URL downloads)
        Loads, chunks, embeds, and uploads to Qdrant
        """
        try:
            logging.info(f" Loading document from: {file_path}")
            
            # Load document
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            
            if not documents:
                return "Failed: No content extracted"
            
            logging.info(f" Loaded {len(documents)} document(s)")
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            logging.info(f" Split into {len(chunks)} chunks")
            
            # Generate embeddings and upload
            logging.info(f" Generating embeddings for {len(chunks)} chunks...")
            
            points = []
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.vectorstore.embeddings.embed_query(chunk.page_content)
                
                # Create point with BOTH 'text' and 'page_content' for compatibility
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        'text': chunk.page_content,
                        'page_content': chunk.page_content,
                        'course_name': course_name,
                        'readable_filename': kwargs.get('readable_filename', 'Unknown'),
                        'url': kwargs.get('url', ''),
                        'base_url': kwargs.get('base_url', ''),
                        'blob_path': kwargs.get('blob_path', ''),
                        'pagenumber': '',
                        'doc_groups': kwargs.get('doc_groups', []),
                    }
                )
                points.append(point)
                
                if (i + 1) % 10 == 0:
                    logging.info(f"   Generated {i + 1}/{len(chunks)} embeddings")
            
            logging.info(f" Generated all {len(points)} embeddings")
            
            # Upload to Qdrant in batches
            batch_size = self.qdrant_upsert_batch_size
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.qdrant_collection_name,
                    points=batch
                )
                logging.info(f"   Uploaded batch {i//batch_size + 1} ({len(batch)} points)")
            
            logging.info(f" Successfully uploaded {len(points)} points to Qdrant")
            
            return "Success"
            
        except Exception as e:
            err_msg = f"Error in _ingest_from_local_file: {str(e)}"
            logging.error(f" {err_msg}")
            logging.error(traceback.format_exc())
            return f"Failed: {err_msg}"

    def split_and_upload(self, texts: List[str], metadatas: List[Dict[str, Any]], force_embeddings: bool, **kwargs):
        """
        This is usually the last step of document ingest. Chunk & upload to Qdrant (and Supabase.. todo).
        Takes in Text and Metadata (from Langchain doc loaders) and splits / uploads to Qdrant.
        """
        logging.info(f"Split and upload invoked with {len(texts)} texts and {len(metadatas)} metadatas")
        # --- PATCH: Normalize doc_groups properly ---
        if 'doc_groups' in kwargs or 'groups' in kwargs:
            doc_groups = kwargs.get('doc_groups') or kwargs.get('groups') or []
            if isinstance(doc_groups, str):
                doc_groups = [doc_groups] if doc_groups.strip() else []
            kwargs['doc_groups'] = doc_groups
            kwargs['groups'] = doc_groups
        else:
            kwargs['doc_groups'] = []
            kwargs['groups'] = []

        
        # Add safety check
        if len(texts) > 1000:
            raise ValueError(f"Too many texts: {len(texts)}. Maximum is 1000.")
        
        # if self.posthog:
        #     self.posthog.capture('distinct_id_of_the_user', event='split_and_upload_invoked',
        #                         properties={
        #                             'course_name': metadatas[0].get('course_name', None),
        #                             'blob_path': metadatas[0].get('blob_path', None),
        #                             'readable_filename': metadatas[0].get('readable_filename', None),
        #                             'url': metadatas[0].get('url', None),
        #                             'base_url': metadatas[0].get('base_url', None),
        #                         })
        assert len(texts) == len(metadatas), f'Text ({len(texts)}) and metadata ({len(metadatas)}) must be equal.'

        try:
            # try to split on paragraphs... fallback to sentences, then chars, ensure we always fit in context window
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            contexts: List[Document] = text_splitter.create_documents(texts=texts, metadatas=metadatas)
            
            # SAFETY: Check chunk count
            if len(contexts) > 2000:
                raise ValueError(f"Too many chunks: {len(contexts)}. Document might be corrupted.")
            
            logging.info(f" Created {len(contexts)} chunks from {len(texts)} texts")
            
            input_texts = [{'input': context.page_content, 'model': self.embedding_model} for context in contexts]

            # Check for duplicates (will also delete data if duplicate is found)
            is_duplicate = self.check_for_duplicates(input_texts, metadatas, force_embeddings)
            if is_duplicate and not force_embeddings:
                    # if self.posthog:
                    #     self.posthog.capture('distinct_id_of_the_user', event='split_and_upload_succeeded',
                    #                         properties={
                    #                             'course_name': metadatas[0].get('course_name', None),
                    #                             'blob_path': metadatas[0].get('blob_path', None),
                    #                             'readable_filename': metadatas[0].get('readable_filename', None),
                    #                             'url': metadatas[0].get('url', None),
                    #                             'base_url': metadatas[0].get('base_url', None),
                    #                             'is_duplicate': True,
                    #                         })
                    logging.info(" Document is duplicate, skipping")
                    return "Success"

            # adding chunk index to metadata for parent doc retrieval
            for i, context in enumerate(contexts):
                context.metadata['chunk_index'] = i

                # ✅ Normalize doc_groups — handle both keys and fallback to metadatas
                groups = (
                    kwargs.get('doc_groups')
                    or kwargs.get('groups')
                    or context.metadata.get('doc_groups')
                    or []
                )
                if isinstance(groups, str) and groups.strip():
                    groups = [groups]
                context.metadata['doc_groups'] = groups


            # ============================================================
            # FIXED: Use Azure OpenAI embeddings directly (not OpenAIAPIProcessor)
            # ============================================================
            logging.info(f" Generating embeddings for {len(contexts)} chunks using Azure OpenAI")
            embeddings_start_time = time.monotonic()
            
            try:
                # Use the vectorstore's embeddings object (which is Azure OpenAI)
                embeddings_dict = {}
                batch_size = 100  # Process in batches to avoid overwhelming the API
                
                for i in range(0, len(contexts), batch_size):
                    batch = contexts[i:i + batch_size]
                    batch_texts = [ctx.page_content for ctx in batch]
                    
                    logging.info(f"   Processing batch {i//batch_size + 1}/{(len(contexts)-1)//batch_size + 1} ({len(batch)} items)")
                    
                    # Generate embeddings using Azure OpenAI
                    batch_embeddings = self.vectorstore.embeddings.embed_documents(batch_texts)
                    
                    # Store in dictionary
                    for text, embedding in zip(batch_texts, batch_embeddings):
                        embeddings_dict[text] = embedding
                    
                    # Small delay between batches to respect rate limits
                    if i + batch_size < len(contexts):
                        time.sleep(5.0)
                
                elapsed_time = time.monotonic() - embeddings_start_time
                logging.info(f" Embeddings generated in {elapsed_time:.2f} seconds ({len(embeddings_dict)} embeddings)")
                
            except Exception as e:
                logging.error(f" Embedding generation failed: {e}")
                logging.error(traceback.format_exc())
                raise

            # Batched upload to Qdrant with temporary indexing threshold adjustments
            collection_name = os.environ['QDRANT_COLLECTION_NAME']  # type: ignore
            
            # Raise indexing threshold to postpone indexing during bulk upserts
            try:
                self.qdrant_client.update_collection(
                    collection_name=collection_name,
                    optimizer_config=models.OptimizersConfigDiff(indexing_threshold=self.qdrant_indexing_threshold_ingest),
                )
            except Exception as e:
                logging.warning("Could not raise Qdrant indexing threshold before ingestion: %s", e)

            try:
                batch: list[PointStruct] = []
                for context in contexts:
                    upload_metadata = {**context.metadata, "page_content": context.page_content}
                    batch.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embeddings_dict[context.page_content],
                            payload=upload_metadata,
                        )
                    )
                    if len(batch) >= self.qdrant_upsert_batch_size:
                        try:
                            self.qdrant_client.upsert(
                                collection_name=collection_name,
                                points=batch,  # type: ignore
                                wait=False,
                            )
                            logging.info(f"   Uploaded batch of {len(batch)} points to Qdrant")
                        except Exception as e:
                            # Timeouts can be acceptable while server processes the request in background
                            logging.warning("Batch upsert encountered an error (continuing): %s", e)
                        finally:
                            batch = []

                if len(batch) > 0:
                    try:
                        self.qdrant_client.upsert(
                            collection_name=collection_name,
                            points=batch,  # type: ignore
                            wait=False,
                        )
                        logging.info(f"   Uploaded final batch of {len(batch)} points to Qdrant")
                    except Exception as e:
                        logging.warning("Final batch upsert encountered an error (continuing): %s", e)
            finally:
                # Revert indexing threshold back to online value
                try:
                    self.qdrant_client.update_collection(
                        collection_name=collection_name,
                        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=self.qdrant_indexing_threshold_online),
                    )
                except Exception as e:
                    logging.warning("Could not revert Qdrant indexing threshold after ingestion: %s", e)

            # Supabase SQL insertion
            contexts_for_supa = [{
                "text": context.page_content,
                "pagenumber": context.metadata.get('pagenumber'),
                "timestamp": context.metadata.get('timestamp'),
                "chunk_index": context.metadata.get('chunk_index'),
                "embedding": embeddings_dict[context.page_content]
            } for context in contexts]
            document = {
                "course_name": contexts[0].metadata.get('course_name'),
                "blob_path": contexts[0].metadata.get('blob_path'),
                "readable_filename": contexts[0].metadata.get('readable_filename'),
                "url": contexts[0].metadata.get('url'),
                "base_url": contexts[0].metadata.get('base_url'),
                "contexts": contexts_for_supa,
            }
            document_size_mb = len(json.dumps(document).encode('utf-8')) / (1024 * 1024)
            logging.info("Inserting document (size: %.2f MB)", document_size_mb)
            insert_status = self.sql_session.insert_document(document)
            if insert_status:
                groups = kwargs.get('groups', '')
                if groups:
                    if contexts[0].metadata.get('url'):
                        count = self.sql_session.add_document_to_group_url(contexts, groups)
                    else:
                        count = self.sql_session.add_document_to_group(contexts, groups)
                    if count == 0:
                        raise ValueError("Error in adding to doc groups")

            # if self.posthog:
            #     self.posthog.capture('distinct_id_of_the_user', event='split_and_upload_succeeded',
            #                         properties={
            #                             'course_name': metadatas[0].get('course_name', None),
            #                             'blob_path': metadatas[0].get('blob_path', None),
            #                             'readable_filename': metadatas[0].get('readable_filename', None),
            #                             'url': metadatas[0].get('url', None),
            #                             'base_url': metadatas[0].get('base_url', None),
            #                             'is_duplicate': False,
            #                         })
            return "Success"
        except Exception as e:
            err: str = f"ERROR IN split_and_upload(): Traceback: {traceback.extract_tb(e.__traceback__)} Error in {inspect.currentframe().f_code.co_name}:{e}"  # type: ignore
            print(err)
            sentry_sdk.capture_exception(e)
            sentry_sdk.flush(timeout=20)
            raise Exception(err)

    def check_for_duplicates(self, texts: List[Dict], metadatas: List[Dict[str, Any]], force_embeddings: bool) -> bool:
        """
        For given metadata, fetch docs from Supabase based on blob path or URL.
        If docs exists, concatenate the texts and compare with current texts, if same, return True.
        """
        course_name = metadatas[0]['course_name']
        incoming_blob_path = metadatas[0]['blob_path']
        url = metadatas[0]['url']

        if incoming_blob_path:
            # Check if uuid (v4) exists in blob_path (not all blob_path have uuids) and remove if necessary
            incoming_filename = incoming_blob_path.split('/')[-1]
            logging.debug(f"Full filename: {incoming_filename}")
            pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}', re.I)
            if bool(pattern.search(incoming_filename)):
                original_filename = incoming_filename[37:]
            else:
                original_filename = incoming_filename
            logging.info(f"Filename after removing uuid: {original_filename}")

            contents = self.sql_session.get_like_docs_by_blob_path(course_name, original_filename)
            contents = contents['data']
            logging.info(
                f"No. of blob path based records retrieved: {len(contents)}")  # multiple records can be retrieved: 3.pdf and 453.pdf
        elif url:
            original_filename = url
            contents = self.sql_session.get_like_docs_by_url(course_name, url)
            contents = contents['data']
            logging.info(f"No. of URL-based records retrieved: {len(contents)}")
        else:
            original_filename = None
            contents = []

        db_whole_text = ""
        exact_doc_exists = False
        if len(contents) > 0:  # a doc with same filename exists in SQL
            for record in contents:
                if incoming_blob_path:
                    curr_filename = record['blob_path'].split('/')[-1]
                    older_blob_path = record['blob_path']
                    if bool(pattern.search(curr_filename)):
                        # uuid pattern exists -- remove the uuid and proceed with duplicate checking
                        sql_filename = curr_filename[37:]
                    else:
                        # do not remove anything and proceed with duplicate checking
                        sql_filename = curr_filename
                elif url:
                    sql_filename = record['url']
                else:
                    continue

                if original_filename == sql_filename:  # compare og blob_path/url with incoming blob_path/url
                    contexts = record
                    exact_doc_exists = True
                    logging.info(f"Exact doc exists in DB: {sql_filename}")
                    break

            if exact_doc_exists:
                for text in contexts['contexts']:
                    db_whole_text += text['text']
                current_whole_text = ""
                for text in texts:
                    current_whole_text += text['input']

                if db_whole_text == current_whole_text:
                    logging.info(f"Duplicate detected: {original_filename}.")
                    if force_embeddings:
                        self.delete_vectors(course_name, older_blob_path, url)
                    return True
                else:
                    print(f"Updated file detected: {original_filename}")
                    if force_embeddings:
                        if incoming_blob_path:
                            delete_status = self.delete_vectors(course_name, older_blob_path, '')
                        else:
                            delete_status = self.delete_vectors(course_name, '', url)
                    else:
                        print("older blob_path/url to be deleted: ", sql_filename)
                        if incoming_blob_path:
                            delete_status = self.delete_data(course_name, older_blob_path, '')
                        else:
                            delete_status = self.delete_data(course_name, '', url)
                return False
            else:
                print(f"NOT a duplicate: {original_filename}")
                return False

        else:
            print(f"NOT a duplicate: {original_filename}")
            return False

    def delete_data(self, course_name: str, blob_path: str, source_url: str):
        """Delete file from blob, Qdrant, and SQL."""
        logging.info(f"Deleting {blob_path} from blob, Qdrant, and SQL for course {course_name}")
        try:
            if blob_path:
                try:
                    blob_client = self.blob_client.get_blob_client(blob_path)
                    blob_client.delete_blob()

                except Exception as e:
                    print("Error in deleting file from blob:", e)
                    sentry_sdk.capture_exception(e)
                # Delete from Qdrant
                # docs for nested keys: https://qdrant.tech/documentation/concepts/filtering/#nested-key
                try:
                    self.qdrant_client.delete(
                        collection_name=self.qdrant_collection_name,
                        points_selector=models.Filter(must=[
                            models.FieldCondition(
                                key="blob_path",
                                match=models.MatchValue(value=blob_path),
                            ),
                        ]),
                    )
                except Exception as e:
                    if "timed out" in str(e):
                        # Timed out still deletes: https://github.com/qdrant/qdrant/issues/3654#issuecomment-1955074525
                        pass
                    else:
                        print("Error in deleting file from Qdrant:", e)
                        sentry_sdk.capture_exception(e)
                        raise e

                try:
                    self.sql_session.delete_document_by_blob_path(course_name=course_name, blob_path=blob_path)
                except Exception as e:
                    print("Error in deleting file from database:", e)
                    sentry_sdk.capture_exception(e)

            # Delete files by their URL identifier
            elif source_url:
                try:
                    self.qdrant_client.delete(
                        collection_name=self.qdrant_collection_name,
                        points_selector=models.Filter(must=[
                            models.FieldCondition(
                                key="url",
                                match=models.MatchValue(value=source_url),
                            ),
                        ]),
                    )
                except Exception as e:
                    if "timed out" in str(e):
                        pass
                    else:
                        print("Error in deleting file from Qdrant:", e)
                        sentry_sdk.capture_exception(e)
                        raise e

                try:
                    self.sql_session.delete_document_by_url(course_name=course_name, url=source_url)
                except Exception as e:
                    print("Error in deleting file from database:", e)
                    sentry_sdk.capture_exception(e)

            return "Success"
        except Exception as e:
            err: str = f"ERROR IN delete_data: Traceback: {traceback.extract_tb(e.__traceback__)} Error in {inspect.currentframe().f_code.co_name}:{e}"  # type: ignore
            sentry_sdk.capture_exception(e)
            return err

    def delete_vectors(self, course_name: str, blob_path: str, source_url: str):
        """Delete vector data from Qdrant and SQL."""
        logging.info(f"Deleting {blob_path} vectors from Qdrant and SQL for course {course_name}")
        try:
            if blob_path:
                # Delete from Qdrant
                # docs for nested keys: https://qdrant.tech/documentation/concepts/filtering/#nested-key
                try:
                    self.qdrant_client.delete(
                        collection_name=self.qdrant_collection_name,
                        points_selector=models.Filter(must=[
                            models.FieldCondition(
                                key="blob_path",
                                match=models.MatchValue(value=blob_path),
                            ),
                        ]),
                    )
                except Exception as e:
                    if "timed out" in str(e):
                        # Timed out still deletes: https://github.com/qdrant/qdrant/issues/3654#issuecomment-1955074525
                        pass
                    else:
                        print("Error in deleting file from Qdrant:", e)
                        sentry_sdk.capture_exception(e)
                        raise e

                try:
                    self.sql_session.delete_document_by_blob_path(course_name=course_name, blob_path=blob_path)
                except Exception as e:
                    print("Error in deleting file from database:", e)
                    sentry_sdk.capture_exception(e)

            # Delete files by their URL identifier
            elif source_url:
                try:
                    self.qdrant_client.delete(
                        collection_name=self.qdrant_collection_name,
                        points_selector=models.Filter(must=[
                            models.FieldCondition(
                                key="url",
                                match=models.MatchValue(value=source_url),
                            ),
                        ]),
                    )
                except Exception as e:
                    if "timed out" in str(e):
                        pass
                    else:
                        print("Error in deleting file from Qdrant:", e)
                        sentry_sdk.capture_exception(e)
                        raise e

                try:
                    self.sql_session.delete_document_by_url(course_name=course_name, url=source_url)
                except Exception as e:
                    print("Error in deleting file from database:", e)
                    sentry_sdk.capture_exception(e)

            return "Success"
        except Exception as e:
            err: str = f"ERROR IN delete_data: Traceback: {traceback.extract_tb(e.__traceback__)} Error in {inspect.currentframe().f_code.co_name}:{e}"  # type: ignore
            sentry_sdk.capture_exception(e)
            return err

    def ingest_single_web_text(self, course_name: str, base_url: str, url: str, content: str, readable_filename: str,
                               force_embeddings: bool, **kwargs) -> Dict[str, None | str | Dict[str, str]]:
        """Crawlee integration"""
        # if self.posthog:
        #     self.posthog.capture('distinct_id_of_the_user', event='ingest_single_web_text_invoked',
        #                          properties={
        #                              'course_name': course_name,
        #                              'base_url': base_url,
        #                              'url': url,
        #                              'content': content,
        #                              'title': readable_filename
        #                          })
        success_or_failure: Dict[str, None | str | Dict[str, str]] = {"success_ingest": None, "failure_ingest": None}
        try:
            metadatas: List[Dict[str, Any]] = [{
                'course_name': course_name,
                'blob_path': '',
                'readable_filename': readable_filename,
                'pagenumber': '',
                'timestamp': '',
                'url': url,
                'base_url': base_url,
            }]
            self.split_and_upload(texts=[content], metadatas=metadatas, force_embeddings=force_embeddings, **kwargs)
            # if self.posthog:
            #     self.posthog.capture('distinct_id_of_the_user',
            #                          event='ingest_single_web_text_succeeded',
            #                          properties={
            #                              'course_name': course_name,
            #                              'base_url': base_url,
            #                              'url': url,
            #                              'title': readable_filename
            #                          })

            success_or_failure['success_ingest'] = url
            return success_or_failure
        except Exception as e:
            err = f" Error in (web text ingest): `{inspect.currentframe().f_code.co_name}`: {e}\nTraceback:\n", traceback.format_exc(
            )  # type: ignore
            print(err)
            sentry_sdk.capture_exception(e)
            success_or_failure['failure_ingest'] = {'url': url, 'error': str(err)}
            return success_or_failure

    def _ingest_single_py(self, blob_path: str, course_name: str, force_embeddings: bool, **kwargs):
        try:
            file_name = blob_path.split("/")[-1]
            file_path = "media/" + file_name  # download from blob to local folder for ingest

            self.blob_client.download_file(self.blob_container, blob_path, file_path)

            loader = PythonLoader(file_path)
            documents = loader.load()

            texts = [doc.page_content for doc in documents]

            metadatas: List[Dict[str, Any]] = [{
                'course_name': course_name,
                'blob_path': blob_path,
                'readable_filename': kwargs.get('readable_filename',
                                                Path(blob_path).name[37:]),
                'pagenumber': '',
                'timestamp': '',
                'url': kwargs.get('url', ''),
                'base_url': kwargs.get('base_url', ''),
            } for doc in documents]
            # print(texts)
            os.remove(file_path)

            success_or_failure = self.split_and_upload(texts=texts, metadatas=metadatas, force_embeddings=force_embeddings, **kwargs)
            print("Python ingest: ", success_or_failure)
            return success_or_failure

        except Exception as e:
            err = f" Error in (Python ingest): `{inspect.currentframe().f_code.co_name}`: {e}\nTraceback:\n", traceback.format_exc(
            )
            print(err)
            sentry_sdk.capture_exception(e)
            return err

    def _ingest_single_vtt(self, blob_path: str, course_name: str, force_embeddings: bool, **kwargs):
        """
        Ingest a single .vtt file from blob.
        """
        try:
            with NamedTemporaryFile(suffix='.vtt', delete=False) as tmpfile:
                # download from blob into vtt_tmpfile
                blob_client = self.blob_client.get_blob_client(blob_path)
                data = blob_client.download_blob().readall()
                tmpfile.write(data)
                tmpfile.flush()
                tmpfile.flush()
                temp_path = tmpfile.name

            try:
                loader = TextLoader(temp_path)
                documents = loader.load()
                texts = [doc.page_content for doc in documents]

                metadatas: List[Dict[str, Any]] = [{
                    'course_name': course_name,
                    'blob_path': blob_path,
                    'readable_filename': kwargs.get('readable_filename',
                                                    Path(blob_path).name[37:]),
                    'pagenumber': '',
                    'timestamp': '',
                    'url': kwargs.get('url', ''),
                    'base_url': kwargs.get('base_url', ''),
                } for doc in documents]

                success_or_failure = self.split_and_upload(texts=texts, metadatas=metadatas, force_embeddings=force_embeddings, **kwargs)
                return success_or_failure
            finally:
                try:
                    os.remove(temp_path)
                except:
                    pass
        except Exception as e:
            err = f" Error in (VTT ingest): `{inspect.currentframe().f_code.co_name}`: {e}\nTraceback:\n", traceback.format_exc()
            print(err)
            sentry_sdk.capture_exception(e)
            return err

    def _ingest_html(self, blob_path: str, course_name: str, force_embeddings: bool, **kwargs) -> str:
        print(f"IN _ingest_html blob_path `{blob_path}` kwargs: {kwargs}")
        try:
            response = self.blob_client.get_object(Bucket=self.blob_container, Key=blob_path)
            raw_html = response['Body'].read().decode('utf-8', errors='ignore')

            soup = BeautifulSoup(raw_html, 'html.parser')
            title = blob_path.replace("courses/" + course_name, "")
            title = title.replace(".html", "")
            title = title.replace("_", " ")
            title = title.replace("/", " ")
            title = title.strip()
            title = title[37:]  # removing the uuid prefix
            text = [soup.get_text()]

            metadata: List[Dict[str, Any]] = [{
                'course_name': course_name,
                'blob_path': blob_path,
                'readable_filename': str(title),  # adding str to avoid error: unhashable type 'slice'
                'url': kwargs.get('url', ''),
                'base_url': kwargs.get('base_url', ''),
                'pagenumber': '',
                'timestamp': '',
            }]

            success_or_failure = self.split_and_upload(text, metadata, force_embeddings, **kwargs)
            print(f"_ingest_html: {success_or_failure}")
            return success_or_failure
        except Exception as e:
            err: str = f"ERROR IN _ingest_html: {e}\nTraceback: {traceback.extract_tb(e.__traceback__)} Error in {inspect.currentframe().f_code.co_name}:{e}"  # type: ignore
            print(err)
            sentry_sdk.capture_exception(e)
            return err

    def _ingest_single_video(self, blob_path: str, course_name: str, force_embeddings: bool, **kwargs) -> str:
        """
        Ingest a single video file from blob.
        """
        print("Starting ingest video or audio")
        try:
            # Ensure the media directory exists
            media_dir = "media"
            if not os.path.exists(media_dir):
                os.makedirs(media_dir)

            # check for file extension
            file_ext = Path(blob_path).suffix
            openai.api_key = os.getenv('OPENAI_API_KEY')
            transcript_list = []
            with NamedTemporaryFile(suffix=file_ext, delete=False) as video_tmpfile:
                # download from blob into an video tmpfile
                self.blob_client.download_fileobj(Bucket=self.blob_container, Key=blob_path, Fileobj=video_tmpfile)
                video_tmpfile.flush()
                temp_path = video_tmpfile.name

            try:
                # try with original file first
                try:
                    mp4_version = AudioSegment.from_file(temp_path, file_ext[1:])
                except Exception as e:
                    print("Applying moov atom fix and retrying...")
                    # Fix the moov atom issue using FFmpeg
                    fixed_video_tmpfile = NamedTemporaryFile(suffix=file_ext, delete=False)
                    try:
                        result = subprocess.run([
                            'ffmpeg', '-y', '-i', temp_path, '-c', 'copy', '-movflags', 'faststart',
                            fixed_video_tmpfile.name
                        ],
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
                        # print(result.stdout.decode())
                        # print(result.stderr.decode())
                    except subprocess.CalledProcessError as e:
                        # print(e.stdout.decode())
                        # print(e.stderr.decode())
                        print("Error in FFmpeg command: ", e)
                        raise e

                    # extract audio from video tmpfile
                    mp4_version = AudioSegment.from_file(fixed_video_tmpfile.name, file_ext[1:])
                    
                    # Clean up fixed file
                    try:
                        os.remove(fixed_video_tmpfile.name)
                    except:
                        pass

                # save the extracted audio as a temporary webm file
                with NamedTemporaryFile(suffix=".webm", dir=media_dir, delete=False) as webm_tmpfile:
                    mp4_version.export(webm_tmpfile, format="webm")
                    webm_temp_path = webm_tmpfile.name

                # check file size
                file_size = os.path.getsize(webm_temp_path)
                # split the audio into 25MB chunks
                if file_size > 26214400:
                    # load the webm file into audio object
                    full_audio = AudioSegment.from_file(webm_temp_path, "webm")
                    file_count = file_size // 26214400 + 1
                    split_segment = 35 * 60 * 1000
                    start = 0
                    count = 0

                    while count < file_count:
                        with NamedTemporaryFile(suffix=".webm", dir=media_dir, delete=False) as split_tmp:
                            if count == file_count - 1:
                                # last segment
                                audio_chunk = full_audio[start:]
                            else:
                                audio_chunk = full_audio[start:split_segment]

                            audio_chunk.export(split_tmp.name, format="webm")
                            split_tmp_path = split_tmp.name

                        # transcribe the split file and store the text in dictionary
                        transcript = self.whisper_client.audio.transcriptions.create(
                            model=self.whisper_deployment,
                            file=f
                        )
                        transcript_list.append(transcript.text) 
                                                
                        # Clean up split file
                        try:
                            os.remove(split_tmp_path)
                        except:
                            pass
                        
                        start += split_segment
                        split_segment += split_segment
                        count += 1
                else:
                    with open(webm_temp_path, "rb") as f:
                        transcript = self.whisper_client.audio.transcriptions.create(
                        model=self.whisper_deployment,
                        file=f
                    )
                    transcript_list.append(transcript.text) 
                # Clean up webm file
                try:
                    os.remove(webm_temp_path)
                except:
                    pass

                text = [txt for txt in transcript_list]
                metadatas: List[Dict[str, Any]] = [{
                    'course_name': course_name,
                    'blob_path': blob_path,
                    'readable_filename': kwargs.get('readable_filename',
                                                    Path(blob_path).name[37:]),
                    'pagenumber': '',
                    'timestamp': text.index(txt),
                    'url': kwargs.get('url', ''),
                    'base_url': kwargs.get('base_url', ''),
                } for txt in text]

                self.split_and_upload(texts=text, metadatas=metadatas, force_embeddings=force_embeddings, **kwargs)
                return "Success"
            finally:
                # Clean up original temp file
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
        except Exception as e:
            err = f" Error in (VIDEO ingest): `{inspect.currentframe().f_code.co_name}`: {e}\nTraceback:\n", traceback.format_exc(
            )
            print(err)
            sentry_sdk.capture_exception(e)
            return str(err)

    def _ingest_single_docx(self, blob_path: str, course_name: str, force_embeddings: bool, **kwargs) -> str:
        try:
            with NamedTemporaryFile(suffix='.docx', delete=False) as tmpfile:
                blob_client = self.blob_client.get_blob_client(blob_path)
                data = blob_client.download_blob().readall()
                tmpfile.write(data)
                tmpfile.flush()

                tmpfile.flush()
                temp_path = tmpfile.name

            try:
                loader = Docx2txtLoader(temp_path)
                documents = loader.load()

                texts = [doc.page_content for doc in documents]
                metadatas: List[Dict[str, Any]] = [{
                    'course_name': course_name,
                    'blob_path': blob_path,
                    'readable_filename': kwargs.get('readable_filename',
                                                    Path(blob_path).name[37:]),
                    'pagenumber': '',
                    'timestamp': '',
                    'url': kwargs.get('url', ''),
                    'base_url': kwargs.get('base_url', ''),
                } for doc in documents]

                self.split_and_upload(texts=texts, metadatas=metadatas, force_embeddings=force_embeddings, **kwargs)
                return "Success"
            finally:
                try:
                    os.remove(temp_path)
                except:
                    pass
        except Exception as e:
            err = f" Error in (DOCX ingest): `{inspect.currentframe().f_code.co_name}`: {e}\nTraceback:\n", traceback.format_exc(
            )
            print(err)
            sentry_sdk.capture_exception(e)
            return str(err)

    def _ingest_single_srt(self, blob_path: str, course_name: str, force_embeddings: bool, **kwargs) -> str:
        try:
            import pysrt

            # NOTE: slightly different method for .txt files, no need for download. It's part of the 'body'
            response = self.blob_client.get_object(Bucket=self.blob_container, Key=blob_path)
            raw_text = response['Body'].read().decode('utf-8', errors='ignore')

            print("UTF-8 text to ingest as SRT:", raw_text)
            parsed_info = pysrt.from_string(raw_text)
            text = " ".join([t.text for t in parsed_info])  # type: ignore
            print(f"Final SRT ingest: {text}")

            texts = [text]
            metadatas: List[Dict[str, Any]] = [{
                'course_name': course_name,
                'blob_path': blob_path,
                'readable_filename': kwargs.get('readable_filename',
                                                Path(blob_path).name[37:]),
                'pagenumber': '',
                'timestamp': '',
                'url': kwargs.get('url', ''),
                'base_url': kwargs.get('base_url', ''),
            }]
            if len(text) == 0:
                return "Error: SRT file appears empty. Skipping."

            self.split_and_upload(texts=texts, metadatas=metadatas, force_embeddings=force_embeddings, **kwargs)
            return "Success"
        except Exception as e:
            err = f" Error in (SRT ingest): `{inspect.currentframe().f_code.co_name}`: {e}\nTraceback:\n", traceback.format_exc(
            )
            print(err)
            sentry_sdk.capture_exception(e)
            return str(err)

    def _ingest_single_excel(self, blob_path: str, course_name: str, force_embeddings: bool, **kwargs) -> str:
        try:
            file_ext = Path(blob_path).suffix
            with NamedTemporaryFile(suffix=file_ext, delete=False) as tmpfile:
                # download from blob into tmpfile
                blob_client = self.blob_client.get_blob_client(blob_path)
                data = blob_client.download_blob().readall()
                tmpfile.write(data)
                tmpfile.flush()

                tmpfile.flush()
                temp_path = tmpfile.name

            try:
                loader = UnstructuredExcelLoader(temp_path, mode="elements")
                documents = loader.load()

                texts = [doc.page_content for doc in documents]
                metadatas: List[Dict[str, Any]] = [{
                    'course_name': course_name,
                    'blob_path': blob_path,
                    'readable_filename': kwargs.get('readable_filename',
                                                    Path(blob_path).name[37:]),
                    'pagenumber': '',
                    'timestamp': '',
                    'url': kwargs.get('url', ''),
                    'base_url': kwargs.get('base_url', ''),
                } for doc in documents]

                self.split_and_upload(texts=texts, metadatas=metadatas, force_embeddings=force_embeddings, **kwargs)
                return "Success"
            finally:
                try:
                    os.remove(temp_path)
                except:
                    pass
        except Exception as e:
            err = f" Error in (Excel/xlsx ingest): `{inspect.currentframe().f_code.co_name}`: {e}\nTraceback:\n", traceback.format_exc(
            )
            print(err)
            sentry_sdk.capture_exception(e)
            return str(err)

    def _ingest_single_image(self, blob_path: str, course_name: str, force_embeddings: bool, **kwargs) -> str:
        try:
            readable_filename = kwargs.get('readable_filename')
            if not readable_filename:
                readable_filename = Path(blob_path).name
                if len(readable_filename) > 37 and readable_filename[36] == '_':
                    readable_filename = readable_filename[37:]
            
            file_extension = Path(readable_filename).suffix or '.png'
            with NamedTemporaryFile(suffix=file_extension, delete=False) as tmpfile:
                # download from blob into tmpfile
                blob_client = self.blob_client.get_blob_client(blob_path)
                data = blob_client.download_blob().readall()
                tmpfile.write(data)
                tmpfile.flush()

                tmpfile.flush()
                temp_path = tmpfile.name
            
            try:
                """
                # Unstructured image loader makes the install too large (700MB --> 6GB. 3min -> 12 min build times). AND nobody uses it.
                # The "hi_res" strategy will identify the layout of the document using detectron2. "ocr_only" uses pdfminer.six. https://unstructured-io.github.io/unstructured/core/partition.html#partition-image
                loader = UnstructuredImageLoader(tmpfile.name, unstructured_kwargs={'strategy': "ocr_only"})
                documents = loader.load()
                """

                doc_groups = kwargs.get('doc_groups', [])
                lang_code = self.get_tesseract_lang_from_docgroups(doc_groups)

                res_str = pytesseract.image_to_string(Image.open(temp_path), lang=lang_code)
                logging.info(f"🧠 Using OCR language '{lang_code}' for doc_groups={doc_groups}")

                print("IMAGE PARSING RESULT:", res_str)
                documents = [Document(page_content=res_str)]

                texts = [doc.page_content for doc in documents]
                metadatas: List[Dict[str, Any]] = [{
                    'course_name': course_name,
                    'blob_path': blob_path,
                    'readable_filename': readable_filename,
                    'pagenumber': '',
                    'timestamp': '',
                    'url': kwargs.get('url', ''),
                    'base_url': kwargs.get('base_url', ''),
                } for doc in documents]

                self.split_and_upload(texts=texts, metadatas=metadatas, force_embeddings=force_embeddings, **kwargs)
                return "Success"
            finally:
                try:
                    os.remove(temp_path)
                except:
                    pass
        except Exception as e:
            err = f" Error in (png/jpg ingest): `{inspect.currentframe().f_code.co_name}`: {e}\nTraceback:\n", traceback.format_exc(
            )
            print(err)
            sentry_sdk.capture_exception(e)
            return str(err)

    def _ingest_single_csv(self, blob_path: str, course_name: str, force_embeddings: bool, **kwargs) -> str:
        try:
            with NamedTemporaryFile(suffix='.csv', delete=False) as tmpfile:
                # download from blob into tmpfile
                blob_client = self.blob_client.get_blob_client(blob_path)
                data = blob_client.download_blob().readall()
                tmpfile.write(data)
                tmpfile.flush()

                tmpfile.flush()
                temp_path = tmpfile.name

            try:
                loader = CSVLoader(file_path=temp_path)
                documents = loader.load()

                texts = [doc.page_content for doc in documents]
                metadatas: List[Dict[str, Any]] = [{
                    'course_name': course_name,
                    'blob_path': blob_path,
                    'readable_filename': kwargs.get('readable_filename',
                                                    Path(blob_path).name[37:]),
                    'pagenumber': '',
                    'timestamp': '',
                    'url': kwargs.get('url', ''),
                    'base_url': kwargs.get('base_url', ''),
                } for doc in documents]

                self.split_and_upload(texts=texts, metadatas=metadatas, force_embeddings=force_embeddings, **kwargs)
                return "Success"
            finally:
                try:
                    os.remove(temp_path)
                except:
                    pass
        except Exception as e:
            err = f" Error in (CSV ingest): `{inspect.currentframe().f_code.co_name}`: {e}\nTraceback:\n", traceback.format_exc(
            )
            print(err)
            sentry_sdk.capture_exception(e)
            return str(err)

    def _ingest_single_pdf(self, blob_path: str, course_name: str, force_embeddings: bool, **kwargs):
        """
        Ingests a single PDF file from MinIO/blob into Qdrant (and optionally Supabase).
        Now Windows-safe (avoids temp file permission issues).
        """

        # NEW LOGIC ADDED
        doc_groups = kwargs.get("doc_groups", []) or kwargs.get("groups", [])
        lang = self.get_tesseract_lang_from_docgroups(doc_groups)

        # Force OCR for Indian languages (Marathi, Hindi, Punjabi, Tamil, Telugu, Gujarati)
        force_ocr = lang in ["mar", "hin", "pan", "tam", "tel", "guj"]

        try:
            readable_filename = kwargs.get('readable_filename', Path(blob_path).name[37:])

            # --- Step 1: Download PDF from blob ---
            with NamedTemporaryFile(suffix='.pdf', delete=False) as tmpfile:
                blob_client = self.blob_client.get_blob_client(blob_path)
                data = blob_client.download_blob().readall()
                tmpfile.write(data)
                tmpfile.flush()

                tmpfile.flush()
                tmp_pdf_path = tmpfile.name

            try:
                # --- Step 2: Read PDF with PyMuPDF ---
                pdf_pages = []
                with fitz.open(tmp_pdf_path) as pdf_document:
                    num_pages = pdf_document.page_count
                    print(f"Processing '{blob_path}' with {num_pages} pages")

                    # Process each page
                    for i, page in enumerate(pdf_document):

                        # *********** NEW: force OCR branch ***********
                        if force_ocr:
                            pix = page.get_pixmap(dpi=300)
                            img_bytes = pix.tobytes("png")
                            img = Image.open(io.BytesIO(img_bytes))
                            text = pytesseract.image_to_string(img, lang=lang)
                        else:
                            text = page.get_text("text")

                        if text.strip():
                            pdf_pages.append({
                                'text': text,
                                'page_number': i + 1,
                                'readable_filename': readable_filename
                            })

                        # --- Step 3: Upload first page thumbnail ---
                        if i == 0:
                            tmp_png_path = os.path.join(
                                os.path.dirname(tmp_pdf_path), f"{Path(blob_path).stem}_thumb.png"
                            )
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # high-res thumbnail
                            pix.save(tmp_png_path)

                            blob_upload_path = (
                                str(Path(blob_path)).rsplit(".pdf")[0] + "-pg1-thumb.png"
                            )

                            with open(tmp_png_path, "rb") as f_png:
                                print("Uploading first-page thumbnail to MinIO...")
                                blob_client = self.blob_client.get_blob_client(blob_upload_path)
                                blob_client.upload_blob(f_png, overwrite=True)

                            try:
                                os.remove(tmp_png_path)
                            except:
                                pass

                # If NOTHING extracted (English case only)
                if not pdf_pages and not force_ocr:
                    print("No text found in PDF, attempting OCR...")
                    return self._ocr_pdf(blob_path, course_name, force_embeddings, **kwargs)

                # --- Step 4: Split and upload ---
                texts = [page['text'] for page in pdf_pages]
                metadatas: List[Dict[str, Any]] = [{
                    'course_name': course_name,
                    'blob_path': blob_path,
                    'readable_filename': readable_filename,
                    'pagenumber': page['page_number'],
                    'timestamp': '',
                    'url': kwargs.get('url', ''),
                    'base_url': kwargs.get('base_url', ''),
                } for page in pdf_pages]

                self.split_and_upload(texts=texts, metadatas=metadatas, force_embeddings=force_embeddings, **kwargs)

                print(f"Successfully ingested: {blob_path}")
                return "Success"

            finally:
                # --- Step 5: Clean up ---
                try:
                    os.remove(tmp_pdf_path)
                except:
                    pass

        except Exception as e:
            err = f"Error in PDF ingest (no OCR): `_ingest_single_pdf`: {e}\nTraceback:\n", traceback.format_exc()
            print(err)
            sentry_sdk.capture_exception(e)
            return str(err)



    def _ocr_pdf(self, blob_path: str, course_name: str, force_embeddings: bool, **kwargs):
        """
        FULL OCR fallback for English PDFs or corrupted PDFs.
        Used only when forced OCR is not triggered inside _ingest_single_pdf.
        """
        pdf_pages_OCRed: List[Dict] = []

        try:
            with NamedTemporaryFile(suffix='.pdf', delete=False) as pdf_tmpfile:
                self.blob_client.download_fileobj(Bucket=self.blob_container, Key=blob_path, Fileobj=pdf_tmpfile)
                pdf_tmpfile.flush()
                temp_path = pdf_tmpfile.name

            try:
                with pdfplumber.open(temp_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        im = page.to_image()
                        doc_groups = kwargs.get('doc_groups', [])
                        lang_code = self.get_tesseract_lang_from_docgroups(doc_groups)

                        # OCR
                        text = pytesseract.image_to_string(im.original, lang=lang_code)
                        logging.info(f"OCR language '{lang_code}' for doc_groups={doc_groups}")

                        print("Page number: ", i, "Text: ", text[:100])
                        pdf_pages_OCRed.append(
                            dict(text=text, page_number=i, readable_filename=Path(blob_path).name[37:])
                        )

                metadatas: List[Dict[str, Any]] = [
                    {
                        'course_name': course_name,
                        'blob_path': blob_path,
                        'pagenumber': page['page_number'] + 1,  # +1 for human indexing
                        'timestamp': '',
                        'readable_filename': kwargs.get('readable_filename', page['readable_filename']),
                        'url': kwargs.get('url', ''),
                        'base_url': kwargs.get('base_url', ''),
                    } for page in pdf_pages_OCRed
                ]

                pdf_texts = [page['text'] for page in pdf_pages_OCRed]

                has_words = any(text.strip() for text in pdf_texts)
                if not has_words:
                    raise ValueError("Failed ingest: No readable text found after OCR.")

                success_or_failure = self.split_and_upload(
                    texts=pdf_texts, metadatas=metadatas, force_embeddings=force_embeddings, **kwargs
                )
                return success_or_failure

            finally:
                try:
                    os.remove(temp_path)
                except:
                    pass

        except Exception as e:
            err = f"Error in PDF ingest (with OCR): `_ocr_pdf`: {e}\nTraceback:\n", traceback.format_exc()
            print(err)
            sentry_sdk.capture_exception(e)
            return err


    def _ingest_single_txt(self, blob_path: str, course_name: str, force_embeddings: bool, **kwargs) -> str:
        """Ingest a single .txt or .md file from blob.
        Args:
            blob_path (str): A path to a .txt file in blob
            course_name (str): The name of the course
        Returns:
            str: "Success" or an error message
        """
        print("In text ingest, UTF-8")
        print("kwargs", kwargs)
        try:
            # NOTE: slightly different method for .txt files, no need for download. It's part of the 'body'
            blob_client = self.blob_client.get_blob_client(blob_path)
            response = blob_client.download_blob().readall()

            text = response.decode('utf-8', errors='ignore')
            print("UTF-8 text to ingest (from blob)", text)
            text = [text]

            metadatas: List[Dict[str, Any]] = [{
                'course_name': course_name,
                'blob_path': blob_path,
                'readable_filename': kwargs.get('readable_filename',
                                                Path(blob_path).name[37:]),
                'pagenumber': '',
                'timestamp': '',
                'url': kwargs.get('url', ''),
                'base_url': kwargs.get('base_url', '')
            }]
            print("Prior to ingest", metadatas)

            success_or_failure = self.split_and_upload(texts=text, metadatas=metadatas,
                                                       force_embeddings=force_embeddings, **kwargs)
            return success_or_failure
        except Exception as e:
            err = f"Error in (TXT ingest): `{inspect.currentframe().f_code.co_name}`: {e}\nTraceback:\n", traceback.format_exc()
            print(err)
            sentry_sdk.capture_exception(e)
            return str(err)

    def _ingest_single_ppt(self, blob_path: str, course_name: str, force_embeddings: bool, **kwargs) -> str:
        """
        Ingest a single .ppt or .pptx file from blob.
        """
        try:
            file_ext = Path(blob_path).suffix
            with NamedTemporaryFile(suffix=file_ext, delete=False) as tmpfile:
                # download from blob into tmpfile
                blob_client = self.blob_client.get_blob_client(blob_path)
                data = blob_client.download_blob().readall()
                tmpfile.write(data)
                tmpfile.flush()

                tmpfile.flush()
                temp_path = tmpfile.name

            try:
                loader = UnstructuredPowerPointLoader(temp_path)
                documents = loader.load()

                texts = [doc.page_content for doc in documents]
                metadatas: List[Dict[str, Any]] = [{
                    'course_name': course_name,
                    'blob_path': blob_path,
                    'readable_filename': kwargs.get('readable_filename',
                                                    Path(blob_path).name[37:]),
                    'pagenumber': '',
                    'timestamp': '',
                    'url': kwargs.get('url', ''),
                    'base_url': kwargs.get('base_url', '')
                } for doc in documents]

                self.split_and_upload(texts=texts, metadatas=metadatas, force_embeddings=force_embeddings, **kwargs)
                return "Success"
            finally:
                try:
                    os.remove(temp_path)
                except:
                    pass
        except Exception as e:
            err = f"Error in (PPTX ingest): `{inspect.currentframe().f_code.co_name}`: {e}\nTraceback:\n", traceback.format_exc(
            )
            print(err)
            sentry_sdk.capture_exception(e)
            return str(err)

    def ingest_github(self, github_url: str, course_name: str, force_embeddings: bool) -> str:
        """
        Clones the given GitHub URL and uses Langchain to load data.
        1. Clone the repo
        2. Use Langchain to load the data
        3. Pass to split_and_upload()
        Args:
            github_url (str): The Github Repo URL to be ingested.
            course_name (str): The name of the course in our system.

        Returns:
            _type_: Success or error message.
        """
        try:
            repo_path = "media/cloned_repo"
            repo = Repo.clone_from(github_url, to_path=repo_path, depth=1, clone_submodules=False)
            branch = repo.head.reference

            loader = GitLoader(repo_path="media/cloned_repo", branch=str(branch))
            data = loader.load()
            shutil.rmtree("media/cloned_repo")
            # create metadata for each file in data

            for doc in data:
                texts = doc.page_content
                metadatas: Dict[str, Any] = {
                    'course_name': course_name,
                    'blob_path': '',
                    'readable_filename': doc.metadata['file_name'],
                    'url': f"{github_url}/blob/main/{doc.metadata['file_path']}",
                    'pagenumber': '',
                    'timestamp': '',
                }
                self.split_and_upload(texts=[texts], metadatas=[metadatas], force_embeddings=force_embeddings)
            return "Success"
        except Exception as e:
            err = f"Error in (GITHUB ingest): `{inspect.currentframe().f_code.co_name}`: {e}\nTraceback:\n{traceback.format_exc()}"
            print(err)
            sentry_sdk.capture_exception(e)
            return err