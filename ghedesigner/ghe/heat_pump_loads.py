import pandas as pd

"""This program will take heating cooling loads input from multiple heat pumps. There is no predetermined heat
extraction or rejection given to the ground rather they are calculated within the simulation."""

# Provide location for your load file here

hp_loads_file = "C:\\Users\\nbast\\Desktop\\Building Loads\\building_loads_SEATTLE_8760_hrs.csv"
hp_constants_file = "C:\\Users\\nbast\\Desktop\\HP_constants\\HP_constants_beta=0.8_SEATTLE.csv"


def get_hp_loads_and_hp_constants(hp_loads_file_path, hp_constants_file_path):
    """
    Reads heat pump loads and heat pump constants data from CSV files.

    Args:
        hp_loads_file_path (str): Path to the CSV file containing heat pump loads.
        hp_constants_file_path (str): Path to the CSV file containing heat pump constants.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - hp_loads_data: DataFrame with heat pump loads.
            - hp_constants_data: DataFrame with heat pump constants.
    """
    hp_loads_data = pd.read_csv(hp_loads_file_path, header=[0], index_col=[0,1,2])
    hp_constants_data = pd.read_csv(hp_constants_file_path, header=[0], index_col=[0,1,2])
    return hp_loads_data, hp_constants_data


hp_loads, hp_constants = get_hp_loads_and_hp_constants(hp_loads_file, hp_constants_file)

#print(hp_loads)
#print(hp_constants)


def calculate_hp_coefficient_values():
    """Calculates the HP coefficient values (r1_hp and r2_hp) based on the given data.

    Returns:
        A tuple containing the calculated r1_hp and r2_hp values.
    """

    r1_hp = 0.0
    r2_hp = 0.0

    for i in range(1, 3):
        for j in range(1, 3):
            r1_value = ((hp_loads.loc[i, j, "HPHtgLd_W"]) * (hp_constants.loc[i, j, "v"]) -
                        (hp_loads.loc[i, j, "HPClgLd_W"]) * (hp_constants.loc[i, j, "b"]))
            r2_value = (((hp_loads.loc[i, j, "HPHtgLd_W"]) * (hp_constants.loc[i, j, "u"]) -
                        (hp_loads.loc[i, j, "HPClgLd_W"]) * (hp_constants.loc[i, j, "a"])) +
                        (hp_loads.loc[i, j, "HtExt_W"]) - (hp_loads.loc[i, j, "HtRej_W"]))

            r1_hp += r1_value
            r2_hp += r2_value

    return r1_hp, r2_hp


r1_hp, r2_hp = calculate_hp_coefficient_values()

r1_hp_array = r1_hp.to_numpy()
r2_hp_array = r2_hp.to_numpy()
#print(r1_hp_array, r2_hp_array)

def extract_time_array():
    """
    Extracts the time values from the specified location in the dataframe
    and converts them into a NumPy array.

    Returns:
    A NumPy array containing the extracted time values.
    """
    time_array = hp_loads.loc[0, 0, "hrs"].to_numpy()
    return time_array


time_hours = extract_time_array()
#print(time_hours)

def time_array_size():
    """
    Extracts the time values from the specified location in the dataframe
    and converts them into a NumPy array.

    Returns:
    Size of NumPy array containing the extracted time values.
    """
    time_array = hp_loads.loc[0, 0, "hrs"].to_numpy()
    size: int = time_array.size
    return size


n = time_array_size()
print(n)



