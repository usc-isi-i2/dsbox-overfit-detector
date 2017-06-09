import unittest
from dsbox.overfitdetector.detector import Detector
from sklearn.linear_model import LogisticRegression, SGDClassifier


class Detectorests(unittest.TestCase):
    def setUp(self):
        self.__detector = Detector()

    def test_detector(self):
        models = [
            {
                'model': SGDClassifier(loss="log", alpha=.0001, n_iter=100),
                'training_instances': [[]],
                'training_labels': []
            },
            {
                'model': LogisticRegression(),
                'training_instances': [[]],
                'training_labels': []
            }
        ]

        reranked = self.__detector.rank_models(models)

        for idx in range(len(reranked)):
            reranked_model = reranked[idx]
            self.assertEqual(reranked_model['rank'], idx+1)
            self.assertEqual(reranked_model['original_index'], idx)
            self.assertEqual(reranked_model['confidence'], 0.0)
if __name__ == '__main__':
    unittest.main()
