from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.experiment.data.training import TrainingData
from arekit.common.model.model_io import BaseModelIO
from arekit.common.opinions.formatter import OpinionCollectionsFormatter
from arekit.contrib.bert.callback import Callback


class CustomTrainingData(TrainingData):

    def __init__(self, labels_count, stemmer, evaluator, opinion_formatter, model_io, callback):
        assert(isinstance(evaluator, BaseEvaluator))
        assert(isinstance(opinion_formatter, OpinionCollectionsFormatter))
        assert(isinstance(model_io, BaseModelIO))
        assert(isinstance(callback, Callback))

        super(CustomTrainingData, self).__init__(labels_count=labels_count,
                                                 stemmer=stemmer)

        self.__evaluator = evaluator
        self.__opinion_formatter = opinion_formatter
        self.__model_io = model_io
        self.__callback = callback

    @property
    def Evaluator(self):
        return self.__evaluator

    @property
    def OpinionFormatter(self):
        return self.__opinion_formatter

    @property
    def ModelIO(self):
        return self.__model_io

    @property
    def Callback(self):
        return self.__callback
