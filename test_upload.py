import os
import requests

# Define the directory containing images
directory = "/Users/vahid/Documents/Pumpkin_Split_Dataset/train/Mosaic Disease"

# List all files inside the directory
files = os.listdir(directory)
print("Files in the directory:", files)

# Select an actual image file (Replace 'sample_image.jpg' with an actual filename from the list)
file_path = os.path.join(directory, "sample_image.jpg")  # Update with a real image name

# Check if the file exists before uploading
if os.path.isfile(file_path):
    print("✅ File exists! Ready to upload.")

    # Define the API endpoint
    url = "http://127.0.0.1:5000/predict"

    # Open and send the image file
    with open(file_path, "rb") as img:
        files = {"file": img}
        response = requests.post(url, files=files)

    # Print the response from the server
    print("Server Response:", response.json())

else:
    print("❌ Error: File not found. Check the path and filename.")
