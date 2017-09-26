import unittest
import os
import random
import pandas as pd
import numpy as np
from dsbox.overfitdetector.detector import Detector
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import logging
import math


class Detectorests(unittest.TestCase):
    def setUp(self):
        self.__detector = Detector()
        self.__dir_path = os.getcwd()
        logging.basicConfig(level=logging.DEBUG)

        self.__train_data_file = self.__dir_path+"/tests/dsbox/overfit-detector/test_data/trainData.csv.gz"
        self.__train_labels_file = self.__dir_path+"/tests/dsbox/overfit-detector/test_data/trainTargets.csv.gz"

        datas = {
            "indep1": [],
            "indep2": [],
            "dep1": [],
            "dep2": [],
            "all_ones": []
        }

        self.__labels = []

        rows = 1000
        for i in range(rows):
            datas["indep1"].append(random.random() * 100)
            datas["indep2"].append(random.random() ** 2)
            datas["all_ones"].append(1)
            if i < rows / 3:
                datas["dep1"].append("A")
                datas["dep2"].append("B")

                self.__labels.append("FOO")
            else:
                if i < rows / 3 * 2:
                    datas["dep1"].append("X")
                    datas["dep2"].append("Y")
                else:
                    datas["dep1"].append("T")
                    datas["dep2"].append("S")

                self.__labels.append("BAR")

        self.__test_df = pd.DataFrame(datas)

    def test_detector(self):

        #data = np.array([1., 2., 3., 4.])
        #labels = np.array(1)
        #for i in range(10):
        #    data = np.vstack([data, [1., 2., 3., 4.]])
        #    labels = np.append(labels, 1)
        #for i in range(10):
        #    data = np.vstack([data, [2., 3., 4., 5.]])
        #    labels = np.append(labels, 0)

        data = pd.read_csv(self.__train_data_file, header=0).fillna(0.0).replace('', '0')
        del data['d3mIndex']
        labels = pd.read_csv(self.__train_labels_file, header=0).fillna(0.0).replace('', '0')['Hall_of_Fame']

        # Encode the categorical data in training data
        # Encode the categorical data in the test targets, uses the first target of the dataset as a target
        trainDataLabelEncoders = dict()
        for col in ['Player', 'Position']:
            trainDataLabelEncoders[col] = preprocessing.LabelEncoder().fit(data[col])
            data[col] = trainDataLabelEncoders[col].transform(data[col])

        # Train the model
        mdl = LogisticRegression().fit(data, labels)
        dd = Detector(n_sample_instances=10, n_sample_iterations=20, columns=['0', '1', '2', '3'], model=mdl)
        dd.set_training_data(inputs=data, outputs=labels)
        dd.set_logger(logging.INFO)

        score = dd.produce(inputs=data)
        self.assertTrue(math.isnan(score))

        dd.fit()
        self.assertTrue(math.isnan(dd.get_confidence()))

    def test_column_independence(self):

        dependent_columns = self.__detector.column_dependencies(self.__test_df)

        self.assertEqual(len(dependent_columns), 1)
        self.assertEqual(dependent_columns[0], ('dep1', 'dep2'))

        column_distributions = dict([(column, self.__detector.column_value_distribution(self.__test_df[column])) for column in self.__test_df.columns])
        column_distributions[dependent_columns[0]] = self.__detector.dependent_columns_value_distribution(self.__test_df['dep1'], self.__test_df['dep2'])
        print(column_distributions['all_ones'])
        self.assertEqual(column_distributions['all_ones'][0], (1, 1.0)) # should be 100% for value 1

        # these should all round to 0.3
        dep_dist = column_distributions[("dep1", "dep2")]
        print(dep_dist)
        for (val, prob) in dep_dist:
            self.assertEqual(round(prob, 1), 0.3)


        # kind of hard to test the shuffling, but we can look at soem properties
        # for instance, the indep and dep columns should still be, and the all_ones should still be all ones
        (shuffled_df, labels_for_shuffle) = self.__detector.shuffle_relabel_inputs(self.__test_df, self.__labels, num_instances=10)

        # it should be that the column all_ones is still all ones
        column_distributions_shuffle = dict([(column, self.__detector.column_value_distribution(shuffled_df[column])) for column in shuffled_df.columns])
        self.assertEqual(column_distributions_shuffle['all_ones'][0], (1, 1.0))  # should be 100% for value 1

        dependent_columns_shuffle = self.__detector.column_dependencies(shuffled_df)
        # and the dependent and independent should still be the same (e.g. it should be true to distribution)
        self.assertEqual(len(dependent_columns_shuffle), 1)
        self.assertEqual(dependent_columns_shuffle[0], ('dep1', 'dep2'))

        # the labels for shuffle should reflect the underlying distribution too
        # in this case, if the shuffled row contains A it should be FOO, and BAR otherwise...
        for idx, row in shuffled_df.iterrows():
            lbl_for_idx = labels_for_shuffle[idx]

            if row.dep1 == 'A':
                self.assertEqual(lbl_for_idx, 'FOO')
            else:
                self.assertEqual(lbl_for_idx, 'BAR')


if __name__ == '__main__':
    unittest.main()
