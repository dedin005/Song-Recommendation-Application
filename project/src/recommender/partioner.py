import os
import shutil

def partition_directory(src_dir, num_partitions):
    """
    Partitions a directory into subdirectories, each containing approximately an equal number of files.

    :param src_dir: Source directory containing the files to be partitioned.
    :param num_partitions: Number of partitions (subdirectories) to create.
    :return: None
    """
    # List all files in the source directory
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    
    # Calculate the number of files per partition
    files_per_partition = len(files) // num_partitions

    # Create and populate the partitions
    for i in range(num_partitions):
        # Define the partition directory name
        partition_dir = os.path.join(src_dir, f'part{i + 1}')

        # Create the partition directory if it doesn't exist
        if not os.path.exists(partition_dir):
            os.makedirs(partition_dir)

        # Determine the files to move to this partition
        start_index = i * files_per_partition
        end_index = None if i == num_partitions - 1 else (i + 1) * files_per_partition
        partition_files = files[start_index:end_index]

        # Move files to the partition directory
        for file in partition_files:
            shutil.move(os.path.join(src_dir, file), os.path.join(partition_dir, file))

# Example usage
src_directory = '../../data/embeddings/parts31to46'
partition_directory(src_directory, 3)
