#!/bin/bash
# Download all objects under a folder prefix from a public GCS bucket.
# Usage: ./gcs_folder_download.sh BUCKET_NAME FOLDER_PREFIX

BUCKET="datadetox"
PREFIX="model_doc/"

if [ -z "$BUCKET" ] || [ -z "$PREFIX" ]; then
  echo "Usage: $0 BUCKET_NAME FOLDER_PREFIX"
  echo "Example: $0 datadetox model_doc/"
  exit 1
fi

echo "Listing objects from gs://$BUCKET/$PREFIX ..."

# Use GCS REST API to list all files in the prefix
curl -s "https://storage.googleapis.com/storage/v1/b/$BUCKET/o?prefix=$PREFIX" \
| grep -o '"name": "[^"]*"' | cut -d'"' -f4 \
| while read -r name; do
    # Skip if it's just the prefix itself
    if [[ "$name" == */ ]]; then
        continue
    fi

    # subfolders locally
    local_path="./data/$name"
    mkdir -p "$(dirname "$local_path")"

    # Download each file
    echo "Downloading: $name"
    curl -s -o "$local_path" "https://storage.googleapis.com/$BUCKET/$name"
done

echo "Done! All files from gs://$BUCKET/$PREFIX downloaded."
