__author__ = "solivr"
__license__ = "GPL"

from typing import List, Union
import csv
import json


def get_hangul_units_form_csv(csv_filename: str) -> List[str]:
    with open(csv_filename, 'r', encoding='utf8') as f:
        csvreader = csv.reader(f, delimiter='\n')
        hangul_units = [row[0] for row in csvreader]
    return hangul_units


def make_json_lookup_hangul(string_chars: str=None, csv_filenames: Union[List[str], str]=None) -> dict:
    """
    :param string_chars: for example string.ascii_letters, string.digits
    :param csv_filenames: csv files containing chars or words in each line.
                    Each line will be considered as a unit in the hangul
    :return:
    """
    lookup = dict()
    offset = 0
    if string_chars:
        # Add characters to lookup table
        lookup.update({char: ord(char) for char in string_chars})
        # Add offset to the codes of hanguls units
        offset = max(lookup.values()) + 1

    if isinstance(csv_filenames, list):
        for file in csv_filenames:
            # Update lookup table with hanguls units from csv file
            hangul_units = get_hangul_units_form_csv(file)
            lookup.update({abbrev: offset + i for i, abbrev in enumerate(hangul_units)})

            # Update offset
            offset = max(lookup.values()) + 1

    elif isinstance(csv_filenames, str):
        hangul_units = get_hangul_units_form_csv(csv_filenames)
        lookup.update({abbrev: offset + i for i, abbrev in enumerate(hangul_units)})

    return map_lookup(lookup)


def load_lookup_from_json(json_filenames: Union[List[str], str])-> dict:
    """
    Load a lookup table from a json file to a dictionnary
    :param json_filenames: either a filename or a list of filenames
    :return:
    """

    lookup = dict()
    if isinstance(json_filenames, list):
        for file in json_filenames:
            with open(file, 'r', encoding='utf8') as f:
                data_dict = json.load(f)
            lookup.update(data_dict)

    elif isinstance(json_filenames, str):
        with open(json_filenames, 'r', encoding='utf8') as f:
            lookup = json.load(f)

    return map_lookup(lookup)


def map_lookup(lookup_table: dict, unique_entry: bool=True)-> dict:
    """
    Converts an existing lookup table with minimal range code ([0, len(lookup_table)-1])
    and avoids multiple instances of the same code label (bijectivity)
    :param lookup_table: dictionary to be mapped {hangul_unit : code label}
    :param unique_entry: If each hangul unit has a unique code and each code a unique hangul unique ('bijective'),
                        only True is implemented for now
    :return: a mapped dictionary
    """

    # Create tuple (hangul unit, code)
    tuple_char_code = list(zip(list(lookup_table.keys()), list(lookup_table.values())))
    # Sort by code
    tuple_char_code.sort(key=lambda x: x[1])

    # If each hangul unit has a unique code and each code a unique hangul unique ('bijective')
    if unique_entry:
        mapped_lookup = [[tp[0], i] for i, tp in enumerate(tuple_char_code)]
    else:
        raise NotImplementedError

    return dict(mapped_lookup)