import pandas as pd
from pathlib import Path
from ghedesigner.manager import run_manager_from_cli_worker


def getting_cop_values():
    """this function reads values of heating and cooling COPs from json file. These values will be later used
    in processing the HP heating loads to heat extraction from the ground and HP cooling loads to heat rejection
    to the ground"""

    with open("C:\\Users\\nbast\\GHEDesigner\\ghedesigner\\my_files\\near_square_for_cli_worker.json", "r") as f:
        input_file = json.load(f)

        cooling_cop = input_file["hp_cop_values"]["hp_cooling_cop"]
        heating_cop = input_file["hp_cop_values"]["hp_heating_cop"]

    return cooling_cop, heating_cop


hp_cooling_cop, hp_heating_cop = getting_cop_values()

# Reading excel/csv file
"""The Excel file is such that it has building index, which may have one or multiple load index and each load
index has four values: HP heating loads, HP cooling loads, heat rejection and heat extraction."""

data = pd.read_csv("C:\\Users\\nbast\\Desktop\\Building_loads\\user_building_loads.csv", header=[0, 1, 2])
total_rows = data.shape[0]
total_columns = data.shape[1]


def total_heating_loads():
    """Each load index has a column for HP heating loads. Each element of the column from all load indices are
    added to form a single column representing total heating loads. The header in Excel/csv file for heating
    loads is named as HPHTgLd_W"""

    """ This part of code does this:
        1. Finds all the columns named "HtExt_W" from level 2 as "HtExt_W" is in level 2 of excel file
        2. finds all matching columns with the third level (2) header "HtExt_W"
        3. prints the matching columns
        4. calculates total heat rejection, i.e. adds each elements in all matching columns and creates
        a resultant summed column. axis=1 means that the addition is along the row of matched columns
        Similarly we can add all other columns
        """

    hp_heating_loads = data.columns.get_level_values(2) == "HPHTgLd_W"
    matching_columns = data.columns[hp_heating_loads]
    #print(matching_columns)  # showing all the matched columns

    hp_heating_loads_sum = data[matching_columns].sum(axis=1)
    heating_loads = hp_heating_loads_sum

    return heating_loads


total_hp_heating_loads = total_heating_loads()

def total_cooling_loads():
    """Each load index has a column for HP cooling loads. Each element of the column from all load indices are
    added to form a single column representing total cooling loads. The header in Excel/csv file for cooling
    loads is named as HPClgLd_W

    How code works is explained in def total_heating_loads"""

    hp_cooling_loads = data.columns.get_level_values(2) == "HPClgLd_W"
    matching_columns = data.columns[hp_cooling_loads]
    #print(matching_columns)  # showing all the matched columns

    hp_cooling_loads_sum = data[matching_columns].sum(axis=1)
    cooling_loads = hp_cooling_loads_sum

    return cooling_loads


total_hp_cooling_loads = total_cooling_loads()


def total_heat_rejection():
    """Each load index has a column for heat rejection. Each element of the column from all load indices are
    added to form a single column representing total heat rejection. The header in Excel/csv file for cooling
    loads is named as HtRej_W

    How code works is explained in def total_heating_loads"""

    heat_rejection = data.columns.get_level_values(2) == "HtRej_W"
    matching_columns = data.columns[heat_rejection]
    #print(matching_columns)  # showing all the matched columns

    heat_rejection_sum = data[matching_columns].sum(axis=1)
    heat_rejection = heat_rejection_sum

    return heat_rejection


total_heat_rejection = total_heat_rejection()


def total_heat_extraction():
    """Each load index has a column for heat extraction. Each element of the column from all load indices are
        added to form a single column representing total heat extraction. The header in Excel/csv file for cooling
    loads is named as HtExt_W

    How code works is explained in def total_heating_loads"""

    heat_extraction = data.columns.get_level_values(2) == "HtExt_W"
    matching_columns = data.columns[heat_extraction]
    #print(matching_columns)  # showing all the matched columns

    heat_extraction_sum = data[matching_columns].sum(axis=1)
    heat_extraction = heat_extraction_sum

    return heat_extraction


total_heat_extraction = total_heat_extraction()


def processing_to_single_load_sequence():
    """this function process loads obtained from above functions into single hourly load sequence, which is used
    as input to GHEDesigner"""
    hp_heating_loads = total_hp_heating_loads
    hp_cooling_loads = total_hp_cooling_loads
    heat_rejection = total_heat_rejection
    heat_extraction = total_heat_extraction

    calc_net_load = ((heat_extraction - heat_rejection) +
                     (hp_heating_loads * (1 - 1 / hp_heating_cop)) - (hp_cooling_loads * (1 + 1 / hp_cooling_cop)))

    return calc_net_load


net_load = processing_to_single_load_sequence()  # assigning the return value from function to net_load
net_loads_list = net_load.tolist()  # converting net_load into list
print(len(net_loads_list))

# reading flag in json file to decide whether to choose data from CSV file or from json file as ground loads.
def json_flag_for_reading_csv_file():
    with open("C:\\Users\\nbast\\GHEDesigner\\ghedesigner\\my_files\\near_square_for_cli_worker.json", "r") as f:
        input_file = json.load(f)
        read_flag_for_csv_file = input_file["loads"]["read_flag_for_csv_file"]

    if read_flag_for_csv_file:
        chosen_ground_loads = net_loads_list
    else:
        chosen_ground_loads = (input_file["loads"]["ground_loads"])

    return chosen_ground_loads


ground_loads = json_flag_for_reading_csv_file()  # assigning the return value of the function to ground_loads


# calling run_manager_for_cli_worker to run the simulation
def main():
    input_filename = Path("C:\\Users\\nbast\\GHEDesigner\\ghedesigner\\my_files\\near_square_for_cli_worker.json")
    output_directory = Path("C:\\Users\\nbast\\GHEDesigner\\ghedesigner\\my_files\\outputs")
    ground_loads_list = ground_loads

    # call run_manager_from_cli_worker function
    run_manager_from_cli_worker(input_filename, output_directory, ground_loads_list)


# Run the main function if script is executed directly
if __name__ == "__main__":
    main()

"""this part of code is for writing the ground_loads_list in text file separated by commas, I did this to generate
list of loads that could be copy-pasted into rowwise_file_for_cli_worker.json. I did this just to check.
this part of program is not necessary and you can generate ground_loads_list without this part of code."""
"""with open("my_list.txt", "w") as f:
    for i, item in enumerate(ground_loads_list):
        f.write(str(item))
        if i < len(ground_loads_list) - 1:
            f.write(", ")
        f.write("\n")"""