from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.experiment.data.training import TrainingData
from arekit.common.model.model_io import BaseModelIO
from arekit.common.opinions.formatter import OpinionCollectionsFormatter


class CustomTrainingData(TrainingData):

    def __init__(self, labels_scaler, stemmer, evaluator, opinion_formatter, model_io):
        assert(isinstance(evaluator, BaseEvaluator))
        assert(isinstance(opinion_formatter, OpinionCollectionsFormatter))
        assert(isinstance(model_io, BaseModelIO))

        super(CustomTrainingData, self).__init__(labels_scaler=labels_scaler,
                                                 stemmer=stemmer)

        self.__evaluator = evaluator
        self.__opinion_formatter = opinion_formatter
        self.__model_io = model_io

    @property
    def Evaluator(self):
        return self.__evaluator

    @property
    def OpinionFormatter(self):
        return self.__opinion_formatter

    @property
    def ModelIO(self):
        return self.__model_io
