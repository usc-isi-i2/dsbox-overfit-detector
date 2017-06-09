"""
This is the detector class. Specifically, it has one entry point, the rank_models method, which takes as input a set
of models, and produces an output that ranks those models, along with associated meta-data.

NOTE: This code is currently stub-code and does nothing. It is written so we can check it into the overall D3M
repository.
"""


class Detector(object):

    def __init__(self):
        pass

    """
    This method ranks models according to how well they generalize (e.g., do not overfit). The overall idea is to test
    each model against a large number of trials of permuted inputs, and measure how the performance changes.

    This focuses on classification and regression problems.

    Parameters
    ----------
    model_list : list
        A list of dicts, where each dict is of the form:
        {
            'model': sklearn model,
            'training_instances': array_like, where each row is one training instance and each column is a feature,
            'training_labels': list, where each index represents the class label (or final output number)
            in correspondence to the rows of the 'training_instances'
        }

        (e.g., [{'model': model, 'training_instances': instances, 'training_labels': labels}, ... , ])

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
    def rank_models(self, model_list):
        model_ranks = []
        for model_info_idx in range(len(model_list)):
            model_info = model_list[model_info_idx]
            confidence = self.__evaluate_model(model_info['model'],
                                               model_info['training_instances'],
                                               model_info['training_labels'])
            model_ranks.append({'rank': model_info_idx+1, 'original_index': model_info_idx, 'confidence': confidence})

        return model_ranks

    """
    This method is internal and is used to evalate the model and produce a confdience value as to how overfit it is,
    or how well it generalizes. It does this by permuting the inputs and comparing the model performance

    Parameters
    ----------
    model : sklearn model
        The model that are you evaluating
    training_data : array_like
        where each row is one training instance and each column is a feature,
    training_labels : list
        where each index represents the class label (or final output number) in correspondence to the rows of the
        'training_instances'
    num_trials: int, optional
        This is the number of trials you will shuffle the inputs and evaluate the model. Defaults to 1000

    Returns
    -------
    double
        the evaulation score representing how overfit or generalized the model is (TBD)

    """
    def __evaluate_model(self, model, training_data, training_labels, num_trials=1000):
        for i in range(num_trials):
            (shuffled_instances, shuffled_labels) = self.__shuffle_relabel_inputs(training_data, training_labels)
        return 0.0

    """
    This method performs a smart shuffling of the input data, keeping the labels appropriately aligned with the
    output data.

    To be clear, we are permuting and shuffling the inputs, so the labels may be based on nearest neighbors, and not
    necessarily provided by the inputs. That is, we may create a permuted instance that is a random selection of
    attributes from a training instance, and therefore the label for htis new instance will be something like the
    most common class of the nearest neighbors to this particular instance, since there is no such instance we have
    ever seen before in our labeled data.

    Parameters
    ----------
    training_data : array_like
        where each row is one training instance and each column is a feature,
    training_labels : list
        where each index represents the class label (or final output number) in correspondence to the rows of the
        'training_instances'

    Returns
    -------
    tuple (array_like, list)
        Returns a tuple where the first item is the shuffled training instances, and the second item are the labels for
        those instances. Note that they are still aligned so the Nth item of the shuffled labels is the classification
        label for the Nth entry in the shuffled training instances

    """
    def __shuffle_relabel_inputs(self, training_data, training_labels):
        return (training_data, training_labels)