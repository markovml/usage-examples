import os
import shutil

def copy_files(local_folder_path: str, git_folder_path: str):
    # Expand `~` in the path to the full directory
    local_folder_path = os.path.expanduser(local_folder_path)
    git_folder_path = os.path.expanduser(git_folder_path)

    # Check if source folder exists
    if not os.path.exists(local_folder_path):
        print(f"Source folder does not exist: {local_folder_path}")
        return

    # Create the destination folder if it does not exist
    if not os.path.exists(git_folder_path):
        os.makedirs(git_folder_path)

    # Iterate through the files in the source folder
    for item in os.listdir(local_folder_path):
        source_path = os.path.join(local_folder_path, item)
        destination_path = os.path.join(git_folder_path, item)

        # Copy only files
        if os.path.isfile(source_path):
            print(f"Copying {source_path} to {destination_path}")
            shutil.copy2(source_path, destination_path)  # Use copy2 to preserve metadata
        else:
            print(f"Skipping {source_path}, as it is not a file.")

if __name__ == "__main__":
    local_folder_path = "~/Downloads/Financial Document Image Dataset/Utility"
    git_folder_path = "/Users/ashita/PycharmProjects/usage-examples/doc_ai_examples/utility_bills" # change folder name
    copy_files(local_folder_path, git_folder_path)
