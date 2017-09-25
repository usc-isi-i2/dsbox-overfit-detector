"""
This is the detector class. It inherits from SupervisedLearnerPrimitiveBase and is written for Python 3.6 now

PLEASE NOTE: This code is work in progress for our year 1 SOW objective. So right now the produce method returns
a np.NaN as a result, so you should not rely on its use.

Because it is not quite a Supervised method, we have adapted its use to fit this API.

The class takes as input the following, class specific parameters:
n_sample_instances: the number of random instances to generate, per iteration, to test model generality
n_samplie_iterations: The number of iterations to run, generating a random data set each iteration. Should be >= 10
columns: A list of strings for column names, helps for debugging
model: A SupervsiedLearningPrimitiveBase object. This is the specific model you will test for overfitting. There are
two major assumptions with this class:

1. The input model is already fit and tuned
2. The data you pass as Sequence[Inputs], either through the produce() method or set_training_data() is already totally
preprocessed through the pipeline. That is, the data is the same as you used to fit() the model you pass as input, since
this class will assume it is already pre-processed
"""

import pandas
import numpy as np
from scipy.stats import chi2_contingency, normaltest
from collections import Counter
import itertools
import uuid
import sklearn.metrics
import random
import logging

from typing import Sequence
from primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

Input = list   # Each row is an array, with the columns as attributes
Output = int   # the final label for that instance
Params = None


class Detector(SupervisedLearnerPrimitiveBase[Input, Output, Params]):

    def __init__(self, *, n_sample_instances: int = 500, n_sample_iterations: int = 100, columns: list = list(),
                 model: SupervisedLearnerPrimitiveBase = None) -> None:
        super().__init__()
        logging.basicConfig(level=logging.INFO)
        self.__logger = logging.getLogger(__name__)
        self.n_sample_iterations = n_sample_iterations
        self.n_sample_instances = n_sample_instances
        self.model = model
        self.data_columns = columns
        self.training_inputs = None
        self.training_outputs = None
        self.fitted = False
        self.confidence = None

    def set_logger(self, lvl):
        self.__logger.setLevel(lvl)

    def set_training_data(self, *, inputs: Sequence[Input], outputs: Sequence[Output]) -> None:
        self.training_inputs = inputs
        self.training_outputs = outputs
        self.fitted = False

    def produce(self, *, inputs: Sequence[Input], timeout: float = None, iterations: int = None) -> Sequence[Output]:
        self.training_inputs = inputs
        if iterations is not None:
            self.n_sample_iterations = iterations
        score = self.__evaluate_model(self.model, self.training_inputs, self.training_outputs)
        self.confidence = np.array([score]) # so it matches the signature of produce
        self.fitted = True
        return np.array([score])

    def fit(self, *, timeout: float = None, iterations: int = None):
        if iterations is not None:
            self.n_sample_iterations = iterations
        self.confidence = self.produce(inputs=self.training_inputs, iterations=iterations)
        self.fitted = True

    def get_confidence(self):
        if self.fitted:
            return self.confidence
        else:
            raise Exception("You must first fit with either fit() or produce()")

    def get_params(self):
        return None

    def set_params(self, *, params: Params):
        pass

    """
    @NOTE: FOR NOW THIS METHOD IS NO LONGER USED...We now run this on one model at a time

    This method ranks models according to how well they generalize (e.g., do not overfit). The overall idea is to test
    each model against a large number of trials of permuted inputs, and measure how the performance changes.

    This focuses on classification and regression problems.

    Parameters
    ----------
    model_list : list
        A list of models of sklearn type

    training_instances : array_like, where each row is one training instance and each column is a feature

    training_labels : list, where each index represents the class label (or final output number)
            in correspondence to the rows of the 'training_instances'
    num_trials: int, optional
        This is the number of trials you will shuffle the inputs and evaluate the model. Defaults to 1000

    Important! The row index of each training instance needs to match the row index of the training labels.
    That is, the Nth row of labels is the class label for the Nth row of the training instance data.

    Returns
    -------
    list
        A list of dicts, where each dict is of the form:
        {
            'rank': int, this models rank in the list (in case you don't rely on list ordering) [one based!!!]
            'original_index': int, the index of this model in the input list (so you can find it again) [zero based]
            'confidence': float, the score the model assigns in terms of how overfit the model is. Since we aren't yet
            in the implementation phase, it's undetermined what this value actually represents.
        }

        Note that this list is returned in order, so that the first model in the list is the model that best
        generalized, and the last in the list performed the worst in terms of generelization (e.g, most overfit).

        Ties are broken randomly.

        NOTE! For this stub implementation, the results are simply the input model list, in order

    """
    #def __rank_models(self, model_list, training_instances, training_labels, num_trials=1000):
    #    model_ranks = []
    #    for model_info_idx in range(len(model_list)):
    #        model_info = model_list[model_info_idx]
    #        confidence = self.evaluate_model(model_info['model'],
    #                                           training_instances,
    #                                           training_labels,
    #                                           num_trials=num_trials)
    #        model_ranks.append({'rank': model_info_idx+1, 'original_index': model_info_idx, 'confidence': confidence})
    #
    #    return model_ranks

    """
    This method is internal and is used to evaluate the model and produce a confidence value as to how overfit it is,
    or how well it generalizes. It does this by permuting the inputs and comparing the model performance

    Parameters
    ----------
    model : sklearn model or sklearn.pipeline model (so you can apply your arbitrary features)
        The model that are you evaluating
    training_data : array_like
        where each row is one training instance and each column is a feature,
    training_labels : list
        where each index represents the class label (or final output number) in correspondence to the rows of the
    num_instances : int
        The number of random rows to generate and label as data to test the model against
        'training_instances'
    num_trials: int, optional
        This is the number of trials you will shuffle the inputs and evaluate the model. Defaults to 1000

    Returns
    -------
    double
        the evaluation score representing how overfit or generalized the model is (TBD)

    """
    def __evaluate_model(self, model, training_data, training_labels, metric='f-measure'):
        # compute a metric on the data, save it
        # compute the metric on all of the shuffled data, across all teh trials, compute hte metric
        # gives you a distribution of metric values. See where original value fit, gives a sense of within mean or not
        #https: // ideas.repec.org / a / eee / econom / v178y2014ip1p194 - 206.#html
        #http: // web.stanford.edu / ~cy10 / public / mrobust / Model_Robustness.pdf
        # NOTE: This is not the metric we want to use, just a placeholder idea for now
        metric_vals = []
        for i in range(self.n_sample_iterations):
            (shuffled_instances, shuffled_labels) = \
                self.shuffle_relabel_inputs(training_data, training_labels, num_instances=self.n_sample_instances)
            score = None

            # run the prediction of the model on the shuffled data, using the shuffled labels
            if metric == 'f-measure':
                score = sklearn.metrics.f1_score(shuffled_labels, model.predict(shuffled_instances), average='micro')

            self.__logger.info("Trial %d / %d (score: %f)" % (i, self.n_sample_iterations, score))

            # store the metric that we score
            assert score is not None
            metric_vals.append(score)

        # apply a test of noramlity to these values. (Note, for this test, if pval < 0.05, then it's NOT normal...)
        test_stat, pval = normaltest(metric_vals)

        sig_val = 0.05  # it should be GREATER than this to be considered normal

        # if it's normal, then we can compute a mean/stdev and figure out how far away from the mean our current
        # value would be on the real-test set. Helps determine if we are overfit or not
        # basically, get a z-score for the true training data/labels and see where it falls. If its within a
        # stddev, then model might be robust (or at least data training for it might be robust)
        if pval > sig_val: #normal!
            stat_mean = np.mean(metric_vals)
            stat_std = np.std(metric_vals)
            train_score = sklearn.metrics.f1_score(training_labels, model.predict(training_data), average='micro')
            z_train_score = (train_score - stat_mean) / stat_std
            self.__logger.info("Train metric: %f Distribution Mean: %f Std: %f, So Z is %f" % (train_score,
                                                                                               stat_mean,
                                                                                               stat_std,
                                                                                           z_train_score))
            # for now, we don't do anything
            return np.NaN
            #return abs(z_train_score)
        #else:
        #    self.__logger.error("Did not gerate normal dist of values of metric.")

        return np.NaN

    """
    This method performs a smart shuffling of the input data, keeping the labels appropriately aligned with the
    output data.

    To be clear, we are permuting and shuffling the inputs, so the labels may be based on nearest neighbors, and not
    necessarily provided by the inputs. That is, we may create a permuted instance that is a random selection of
    attributes from a training instance, and therefore the label for this new instance will be something like the
    most common class of the nearest neighbors to this particular instance, since there is no such instance we have
    ever seen before in our labeled data.

    Parameters
    ----------
    training_data : array_like
        where each row is one training instance and each column is a feature,
    training_labels : list
        where each index represents the class label (or final output number) in correspondence to the rows of the
        'training_instances'
    num_instances : int
        The number of random rows to generate and label as data to test the model against

    Returns
    -------
    tuple (array_like, list)
        Returns a tuple where the first item is the shuffled training instances, and the second item are the labels for
        those instances. Note that they are still aligned so the Nth item of the shuffled labels is the classification
        label for the Nth entry in the shuffled training instances

    """
    def shuffle_relabel_inputs(self, training_data, training_labels, num_instances, label_chooser='random_sample'):

        if not isinstance(training_data, pandas.DataFrame):
            num_cols = len(training_data[0])
            cols = ["%dZZ" % z for z in range(num_cols)]  # just need a string
            training_data = pandas.DataFrame(data=training_data, columns=cols)
            # raise Exception("We only support DataFrames for now")

        # these are pairs of columns (A,B) where A and B have some dependent relationship
        dependent_column_pairs = self.column_dependencies(training_data)

        # get a list of columns that appear in the dependent set, just so we can ignore them when creating
        # column value distributions for the independent columns
        dependent_columns = list(itertools.chain(*dependent_column_pairs))


        # now, sample from each independent column, according to it's distribution
        # and for the dependent columns, according to the conditional probabilities

        self.__logger.debug("Dependent columns are: %s" % str(dependent_column_pairs))

        if isinstance(training_data, pandas.DataFrame):
            column_distributions = dict([(column, self.column_value_distribution(training_data[column])) for column in training_data.columns if column not in dependent_columns])

            for (col1, col2) in dependent_column_pairs:
                # get their dependencies
                column_distributions[(col1, col2)] = self.dependent_columns_value_distribution(training_data[col1], training_data[col2])

            # now, for any columns that are independent, we can sample according to the distribution
            # for those that are dependent, we need those distributions...
            shuffled_data_dict = {}
            for cls in column_distributions:

                # could be a single column or a tuple!
                vals = [t[0] for t in column_distributions[cls]]
                probs = [t[1] for t in column_distributions[cls]]

                if isinstance(cls, tuple): # these would be the dependent guys...
                    # we need to remap the tuple to a random string so we can generate the random values appropriately
                    # and don't want to encode the tuple as a string, bc we want to preserve its members
                    vals_remap = dict([(uuid.uuid4(), val) for val in vals])
                    vals_remap_keys = vals_remap.copy().keys()
                    rand_values = np.random.choice(list(vals_remap_keys), size=num_instances, replace=True, p=probs)

                    # might need to split this up, depending on whether its a single column or not
                    col1 = cls[0]
                    col2 = cls[1]
                    col1vals = [vals_remap[x][0] for x in rand_values]
                    col2vals = [vals_remap[x][1] for x in rand_values]
                    shuffled_data_dict[col1] = col1vals
                    shuffled_data_dict[col2] = col2vals
                else:
                    rand_values = np.random.choice(vals, size=num_instances, replace=True, p=probs)
                    shuffled_data_dict[cls] = rand_values

            # now we have our randomized rows! This is the randomly generated data set, true to the distribution
            # of the underlying training data (ish)
            shuffled_df = pandas.DataFrame(shuffled_data_dict)

            self.__logger.debug("RANDOMLY GENERATED RECORDS...\n%s" % str(shuffled_df))

            # given the shuffled data points, we next find the "closest" data points from the input set
            # and use them to assign a label to our new data
            # the way to assign them is to take all of the labels for those closest guys, and create
            # a label distribution. Then you make a random draw, and pick the label accordingly
            # this way, over lots of draws, you will approximate the truth label distribution to those records

            # we need to append the lables to these guys, or else get them as we get the matches...
            # call it zzz_training_labels to minimize chance of conflict. Could use a random string but not sure...
            training_data_with_labels = training_data.assign(zzz_training_labels=pandas.Series(training_labels).values)

            associated_labels = []  # this is the labels picked for each of our randomly generated rows
            for idx, row in shuffled_df.iterrows():
                matches_w_labels = self.find_matching_rows(row.values, shuffled_df.columns, training_data_with_labels)
                # the training labels reflect the distribution over the matching rows
                # so just pick one randomly and that should reflect the prob of seeing that label given the matches
                # we could do it as a weighted prob based on matching more cols, but for now we do this
                all_labels = matches_w_labels['zzz_training_labels']
                if label_chooser == 'random_sample':
                    lbl = random.choice(all_labels.values)
                elif label_chooser == 'max_neighbor':
                    # find the one with the maximum count...
                    lbl, lbl_count = Counter(all_labels.values).most_common(1)[0]
                else:
                    raise Exception("Must be one of the other choices")

                associated_labels.append(lbl)

            return shuffled_df, associated_labels
        return None, None

    def find_matching_rows(self, row_values, columns, training_data, num_neighbors=10, max_depth=2, real_value_extend=0.1):
        # real_value_extend is the amount to extend the search w/in real-valued columns. For instance,
        # without this you would look for all rows where

        # find rows with the most matching columns
        # so, start with those that match all cols (e.g., match exactly)
        # then n-1 cols (which is multiple of them)
        # then n-2, etc.
        # until you've found enough rows, or return those that you've found
        # https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas

        curr_size = len(columns)

        # then you make combinations of decreasing size. Note, these need to be limited in size/scope!!!
        # Maybe we look at N, N-1, N-2 and N-3, but no more!
        stopping_size = len(columns) - max_depth

        while curr_size >= stopping_size:
            all_matches = pandas.DataFrame()

            cols_to_match = itertools.combinations(columns, curr_size)

            for col_set in cols_to_match:
                self.__logger.debug("COL SET: %s" % str(col_set))
                # find the rows of training_data that match the criteria
                # to do this, use the pandas query syntax
                # df.query('foo == 111 & bar == 444')
                qry_string = ''

                # get the values for those specific columns
                vals_for_col_set = []

                for c in col_set:
                    idx = list(columns.values).index(c)
                    val = row_values[idx]
                    vals_for_col_set.append(val)

                matches = training_data
                for (col, val) in zip(col_set, vals_for_col_set):
                    if self.is_number(val):
                        lower_bound = float(val) * (1.0 - real_value_extend)
                        upper_bound = float(val)*(1.0 + real_value_extend)
                        matches = matches[matches[col] >= lower_bound]
                        matches = matches[matches[col] <= upper_bound]
                    else:
                        if val.isalnum():
                            matches = matches[matches[col] == '%s' % val]
                        else:
                            matches = matches[matches[col] == val]

                self.__logger.debug("Dataframe query: %s found %d matches." % (qry_string, len(matches)))
                if len(matches) > 0:
                    all_matches = all_matches.append(matches)

            if len(all_matches) > 0:
                if len(all_matches) > num_neighbors:
                    all_matches = all_matches.sample(frac=1).reset_index(drop=True)  # like a shuffle operation
                self.__logger.debug(str(all_matches[:num_neighbors]))
                return all_matches[:num_neighbors]

            curr_size -= 1

        return pandas.DataFrame()

    """ In order to more appropriately shuffle the data, we need to know which columns are independent from others
    and which are not.

    So, in this way, we can freely sample from independent columns (according to their sample distribution),
    but sample conditionally from those that are conditionally dependent (according to their conditional distribution)

    For now, to keep it simple, we look at pairs (e.g., we don't yet consider n-ary conditional dependence)

    NOTE:  If you have less than 1,000 observations total, then in general that means your chi-square test for
    column independence is going to be unreliable (http://www.biostathandbook.com/small.html). However, it's not
    really feasible to run a Fisher's exact test for the MxN data in that case either, so for now we just assume
    that the columns are independent if you have less than 1,000 observations total.

    Parameters
    ----------
    training_data : DataFrame
        where each row is one training instance and each column is a feature

    Returns
    -------
    TBD

    """
    def column_dependencies(self, training_data, significance_level=0.05):

        if not isinstance(training_data, pandas.DataFrame):
            raise Exception("Currently we only support DataFrames")

        dependent_columns = []

        all_columns = list(training_data)

        for i in range(len(all_columns)):
            for j in range(len(all_columns)):

                if i >= j:
                    continue

                # nice, elegant solution via: https://codereview.stackexchange.com/questions/96761/chi-square-independence-test-for-two-pandas-df-columns
                # for each pair of columns, test them...
                # if they are independent, fine, bc symmetric
                # if not, then we know we will use the conditional probability tables for our sampling later

                first_col = all_columns[i]
                second_col = all_columns[j]

                groupsizes = training_data.groupby([first_col, second_col]).size()
                ctsum = groupsizes.unstack(first_col)

                # fillna(0) is necessary to remove any NAs which will cause exceptions
                data_for_analysis = ctsum.fillna(0)

                #print data_for_analysis.values.sum()
                # pretty good reasoning for using exact less than 1000, and standard chi2 otherwise
                # http://www.biostathandbook.com/small.html
                if data_for_analysis.values.sum() < 1000:
                    # now, its not super feasible and is perhaps not necessary for us to run the Fisher's exact
                    # test, since we don't really care, perhaps. We want to know if things are independent bc
                    # we want to know if we can just sample from them, or if we have to sample conditionally...
                    # so, basically, if you have less than 1000 entries, we just assume all columns are independent
                    pass

                (chi_val, pval) = chi2_contingency(data_for_analysis)[:2]

                if pval <= significance_level:
                    dependent_columns.append((first_col, second_col))

                #print "%s vs %s" % (first_col, second_col)
                #print "Chi: %f PVal: %f" % (chi_val, pval)

        return dependent_columns

    """Takes a column, and returns a distribution of the values (e.g., probability of seeing each value)
    so you can generate a random sampler of that value.

    E.g., you choose a value using: np.random.choice(aa_milne_arr, replace=True, p=[0.5, 0.1, 0.1, 0.3])
    and this method is what creates that last parameter p
    """
    def column_value_distribution(self, column):
        data = column.values
        return self.counts_to_distribution(data)

    """Takes two columns, that have a depdendency, and converts pairs of values in probability of seeing
    that pair
    """
    def dependent_columns_value_distribution(self, col1, col2):
        # basically the same type of distribution you see for single, but with pairs of values
        together = zip(col1, col2)  # now they are aligned by index
        return self.counts_to_distribution(together)

    """
        Does the actual computation of column values to the propbability of seeing that value

    """
    def counts_to_distribution(self, column_values):
        counts = Counter(column_values)
        total = float(sum(counts.values()))

        # these need to be registered in 1-1 for this all to work
        distribution = []
        for colval in counts:
            distribution.append((colval, float(counts[colval]) / total))

        return distribution

    def csv_to_dataframe(self, path_to_csv):
        data = pandas.read_csv(path_to_csv)  # all read as str
        return data

    def is_number(self, s):
        # type: (number) -> number
        try:
            float(s)
            return True
        except ValueError:
            return False
