from fastai.vision.all import *
import pathlib
from cognite.client.data_classes import FileMetadataUpdate

def handle(client, data):
    
    model_path = pathlib.Path('models/model.pkl')
    learn = load_learner(model_path)

    # Define the metadata filter
    metadata_filter = {
        "classify": "true"
    }

    # Retrieve files with the specified metadata filter
    files_with_classify_true = client.files.list(metadata=metadata_filter)

    # Print the file names and IDs
    for file in files_with_classify_true:
        file_metadata = client.files.retrieve(id=file.id)
        file_content = client.files.download_bytes(id=file.id)

        download_path = "fromcdf/"+file_metadata.name

        with open(download_path, "wb") as f:
            f.write(file_content)

        type_of_beverage,_,probs = learn.predict(download_path)

        # Define the new metadata
        new_metadata = {
            "beverage": type_of_beverage  # Add a new metadata key-value pair
        }

        my_update = FileMetadataUpdate(id=file.id).metadata.add({"beverage": type_of_beverage})

        # Update the file with the new metadata
        updated_file = client.files.update(my_update)

        # Print the updated file metadata
        print("Updated metadata:", updated_file.metadata)

        os.remove(download_path)

    return