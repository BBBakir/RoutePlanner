import pandas as pd
import os
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def StructureChange(df,save_path):
    result = pd.DataFrame(columns=['Location', 'Truck', 'Lorry', 'Iveco', 'Bulk'])

    for index, row in df.iterrows():
        name = row['Names']
        value = float(row['Prices'].replace('₺', '').replace('.', '').replace(',', '.'))

        if 'TIR' in name:
            vehicle = 'Truck'
            location = name.replace('TIR', '')
        elif 'KAMYON' in name:
            vehicle = 'Lorry'
            location = name.replace('KAMYON', '')
        elif 'IVECO' in name:
            vehicle = 'Iveco'
            location = name.replace('IVECO', '')

        method = None
        if 'DÖKME' in location:
            method = 'Bulk'
            location = location.replace('DÖKME', '')
        location = location.strip().replace('i', 'İ').upper()
        if location not in result['Location'].values:
            new_row = pd.DataFrame({'Location': [location], 'Truck': [0], 'Lorry': [0], 'Iveco': [0], 'Bulk': [0]})
            if method == 'Bulk':
                new_row['Bulk'] = value
            else:
                new_row[vehicle] = value
            if not new_row.empty and not new_row.isna().all().all():
                result = pd.concat([result, new_row], ignore_index=True)
        else:
            if method == 'Bulk':
                result.loc[result['Location'] == location, 'Bulk'] = value
            else:
                result.loc[result['Location'] == location, vehicle] = value

    result.to_csv(save_path, index=False)
    return result

def ReverseStructureChange(df):
    rows = []

    for index, row in df.iterrows():
        location = row['Location']

        if row['Truck'] != 0:
            rows.append({'Names': 'TIR' + location , 'Prices': row['Truck']})
        if row['Lorry'] != 0:
            rows.append({'Names':  'KAMYON' + location, 'Prices': row['Lorry']})
        if row['Iveco'] != 0:
            rows.append({'Names':  'IVECO' + location, 'Prices': row['Iveco']})
        if row['Bulk'] != 0:
            rows.append({'Names':  'DÖKME' + location, 'Prices': row['Bulk']})

    original_df = pd.DataFrame(rows)
    return original_df

def custom_sort(df):
    # Define a sorting key function
    def sorting_key(row):
        name = row.get('Names', "")
        if 'TIR' in name:
            return (1, name)
        elif 'DÖKME' in name:
            return (2, name)
        elif 'KAMYON' in name:
            return (3, name)
        elif 'IVECO' in name:
            return (4, name)
        else:
            return (5, name)  # for any other case not considered

    # Sort the dataframe based on the custom key
    sorted_rows = sorted(df.iterrows(), key=lambda x: sorting_key(x[1]))

    # Recreate the DataFrame from sorted rows
    sorted_df = pd.DataFrame([row[1] for row in sorted_rows])

    return sorted_df.reset_index(drop=True)

'''
    matrices
'''



def remove_accents(text):
    turkish_to_english = str.maketrans("ğĞıİöÖşŞüÜçÇ", "gGiIoOsSuUcC")
    return text.translate(turkish_to_english)


def process_city_name(city_name):
    if '-' in city_name:
        return city_name.split('-')[0]
    return city_name


def get_cost_from_kutahya(city, data, vechile):
    if city == "kutahya":
        return 0
    if city in ['sincan', 'akyurt', 'akinci']:
        filtered_df = data[data['Location'].str.lower() == 'ankara']
    elif city == 'caycuma':
        filtered_df = data[data['Location'].str.lower() == 'konya']
    elif city == 'mudanya':
        filtered_df = data[data['Location'].str.lower() == 'bursa']
    else:
        filtered_df = data[data['Location'].str.lower() == remove_accents(city).lower()]

    if not filtered_df.empty:
        if vechile =='Bulk':
            return filtered_df.iloc[0]['Truck'] + filtered_df.iloc[0]['Bulk']
        else:
            return filtered_df.iloc[0][vechile]
    else:
        return 0


def compute_costs(city, matrix_df, edge_list_copy, data, vechile):
    i = 0
    while i < len(edge_list_copy):
        src, dest = edge_list_copy[i]
        src = remove_accents(process_city_name(src).lower())
        dest = remove_accents(process_city_name(dest).lower())

        if src == city:

            dest_cost_from_kutahya = get_cost_from_kutahya(dest, data, vechile)
            src_cost_from_kutahya = get_cost_from_kutahya(src, data, vechile)

            relative_cost = dest_cost_from_kutahya - src_cost_from_kutahya
            if relative_cost == 0:
                relative_cost = 10
            matrix_df.at[src, dest] = relative_cost

            edge_list_copy.pop(i)
            compute_costs(dest, matrix_df, edge_list_copy, data, vechile)
            i = 0
        else:
            i += 1  # Only increment i if an item was not removed


def edges_calc(edge_list):
    data = pd.read_csv(resource_path('Assets/Data/Pricing/pricing.csv'), encoding='utf-8')
    data['Location'] = data['Location'].apply(lambda x: remove_accents(x).lower())

    edge_list_copy = [(
        process_city_name(remove_accents(src).lower().capitalize()),
        process_city_name(remove_accents(dest).lower().capitalize())
    ) for src, dest in edge_list]

    vechile_list = ['Truck', 'Lorry', 'Iveco', 'Bulk']
    matrix_df = pd.DataFrame(0, index=data['Location'], columns=data['Location'])
    locations_to_zero = ['kutahya', 'sincan', 'akyurt', 'akinci', 'mudanya', 'caycuma']
    for location in locations_to_zero:
        matrix_df.loc[location] = 0

    for location in locations_to_zero:
        matrix_df[location] = 0

    for vechile in vechile_list:
        compute_costs('kutahya', matrix_df, edge_list_copy, data, vechile)
        matrix_df.to_csv(resource_path(f'Assets/Data/Pricing/{vechile}_matrix.csv'))
