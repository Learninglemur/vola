from google.cloud import storage
import io
from datetime import datetime

# Initialize the client
storage_client = storage.Client()

# Your bucket name
bucket_name = "vola_bucket"

def save_to_bucket_example():
    try:
        # Get bucket
        bucket = storage_client.bucket(bucket_name)

        # Create some dummy data
        dummy_data = "This is some test data\nSecond line\nThird line"
        
        # Create a timestamp for unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"test_file_{timestamp}.txt"

        # Create a blob (object) in the bucket
        blob = bucket.blob(f"test_files/{filename}")

        # Upload the data
        blob.upload_from_string(dummy_data)

        print(f"File saved successfully to: gs://{bucket_name}/test_files/{filename}")
        return True

    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return False

# Test saving a CSV file
def save_csv_example():
    try:
        bucket = storage_client.bucket(bucket_name)
        
        # Create dummy CSV data
        csv_data = "name,age,city\nJohn,30,New York\nJane,25,Los Angeles"
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"test_data_{timestamp}.csv"
        
        # Create blob and upload
        blob = bucket.blob(f"csv_files/{filename}")
        blob.upload_from_string(csv_data, content_type='text/csv')
        
        print(f"CSV saved successfully to: gs://{bucket_name}/csv_files/{filename}")
        return True
        
    except Exception as e:
        print(f"Error saving CSV: {str(e)}")
        return False

# Run the examples
if __name__ == "__main__":
    save_to_bucket_example()
    save_csv_example()
