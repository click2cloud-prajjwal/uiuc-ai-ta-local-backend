import os
from azure.storage.blob import (
    BlobServiceClient,
    generate_blob_sas,
    BlobSasPermissions,
    PublicAccess
)
from datetime import datetime, timedelta


class BlobStorage:
    def __init__(self):
        # Get environment variables
        self.account_name = os.getenv("AZURE_SA_NAME")
        self.account_key = os.getenv("AZURE_SA_ACCESSKEY")
        self.container_name = os.getenv("AZURE_STORAGE_CONTAINER", "uiuc-chatbot")

        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

        if not connection_string:
            connection_string = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={self.account_name};"
                f"AccountKey={self.account_key};"
                f"EndpointSuffix=core.windows.net"
            )

        self.client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.client.get_container_client(self.container_name)

        # Create container if not exists (and make it public)
        try:
            self.container_client.get_container_properties()
        except Exception as e:
            if "ContainerNotFound" in str(e):
                self.client.create_container(
                    self.container_name,
                    public_access=PublicAccess.Blob  # public blobs
                )
                print(f"Created new public container: {self.container_name}")
            else:
                raise e

        # Ensure container is public (even if it existed before)
        try:
            props = self.container_client.get_container_access_policy()
            if not props.get("public_access") or props["public_access"] != "blob":
                self.container_client.set_container_access_policy(public_access=PublicAccess.Blob)
                print(f"Updated container access policy → public (blob)")
        except Exception as e:
            print(f"Could not verify/update container access policy: {e}")
            
    def get_blob_url(self, blob_key: str) -> str:
        return f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob_key}"
        
    def upload_file(self, file_path: str, blob_name: str):
        """Upload local file to Azure Blob"""
        with open(file_path, "rb") as data:
            self.container_client.upload_blob(name=blob_name, data=data, overwrite=True)
        print(f"Uploaded: {blob_name}")

    def download_file(self, blob_name: str, download_path: str):
        """Download blob to local path"""
        blob_client = self.container_client.get_blob_client(blob_name)
        with open(download_path, "wb") as file:
            file.write(blob_client.download_blob().readall())
        print(f"Downloaded: {blob_name} → {download_path}")

    def delete_file(self, blob_name: str):
        """Delete a blob"""
        blob_client = self.container_client.get_blob_client(blob_name)
        blob_client.delete_blob()
        print(f"Deleted: {blob_name}")

    def generate_presigned_url(self, blob_name: str, expiration_hours: int = 1):
        """Generate a temporary read URL for the blob"""
        sas_token = generate_blob_sas(
            account_name=self.client.account_name,
            container_name=self.container_name,
            blob_name=blob_name,
            account_key=self.client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now() + timedelta(hours=expiration_hours),
        )
        return f"https://{self.client.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}?{sas_token}"
