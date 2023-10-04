import pandas as pd
import os
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def read_and_process_matrix(file_path):
    # Step 1: Load the matrix from a CSV file
    df = pd.read_csv(file_path, index_col=0)

    # Step 2: Convert the matrix to a format usable in your VRP solver.
    # Assuming your solver can use a 2D list of distances.
    matrix = df.values.tolist()

    return matrix


def create_mappings(file_path):
    df = pd.read_csv(resource_path(file_path), index_col=0)
    city_to_index = {city: index for index, city in enumerate(df.index)}
    index_to_city = {index: city for index, city in enumerate(df.index)}
    return city_to_index, index_to_city


def create_distance_matrices_dict():
    # File paths to your CSV files
    iveco_file_path = resource_path('Assets/Data/Pricing/Iveco_matrix.csv')
    truck_file_path = resource_path('Assets/Data/Pricing/Truck_matrix.csv')
    bulk_file_path = resource_path('Assets/Data/Pricing/Bulk_matrix.csv')
    lorry_file_path = resource_path('Assets/Data/Pricing/Lorry_matrix.csv')

    # Create mappings from any matrix file as all have the same city index
    city_to_index, index_to_city = create_mappings(iveco_file_path)

    # Process each matrix
    iveco_matrix = read_and_process_matrix(iveco_file_path)
    truck_matrix = read_and_process_matrix(truck_file_path)
    bulk_matrix = read_and_process_matrix(bulk_file_path)
    lorry_matrix = read_and_process_matrix(lorry_file_path)

    # Store each processed matrix in a dictionary under the respective vehicle type key
    distance_matrices = {
        'Iveco': iveco_matrix,
        'Truck': truck_matrix,
        'Bulk': bulk_matrix,
        'Lorry': lorry_matrix
    }

    return distance_matrices, city_to_index, index_to_city


