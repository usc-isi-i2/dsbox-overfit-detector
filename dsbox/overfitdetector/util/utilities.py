import pandas as pd


class Utilities(object):

    def csv_to_dataframe(self, path_to_csv):
        data = pd.read_csv(path_to_csv)  # all read as str
        return data

    @staticmethod
    def is_number(s):
        # type: (number) -> number
        try:
            float(s)
            return True
        except ValueError:
            return False
