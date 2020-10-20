#!/usr/bin/python
import argparse
import logging

from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler
from arekit.contrib.experiments.rusentrel.experiment import RuSentRelExperiment
from arekit.contrib.experiments.rusentrel_ds.experiment import RuSentRelWithRuAttitudesExperiment
from arekit.contrib.bert.encoder import BertEncoder
from args.bert_formatter import BertFormatterArg
from args.cv_index import CvCountArg
from args.experiment import ExperimentTypeArg, SUPERVISED_LEARNING, SUPERVISED_LEARNING_WITH_DS
from args.labels_count import LabelsCountArg
from args.ra_ver import RuAttitudesVersionArg

from io_utils import RuSentRelBasedExperimentsIOUtils


def __encode(experiment, formatter):
    BertEncoder.to_tsv(experiment=experiment,
                       sample_formatter=formatter)


def __cv_based_experiment(experiment, formatter, cv_count=3):
    experiment.DataIO.CVFoldingAlgorithm.set_cv_count(cv_count)
    for cv_index in range(cv_count):
        experiment.DataIO.CVFoldingAlgorithm.set_iteration_index(cv_index)
        __encode(experiment=experiment, formatter=formatter)


def __non_cv_experiment(experiment, formatter):
    __cv_based_experiment(experiment=experiment,
                          formatter=formatter,
                          cv_count=1)


def create_labels_scaler(labels_count):
    assert (isinstance(labels_count, int))

    if labels_count == 2:
        return TwoLabelScaler()
    if labels_count == 3:
        return ThreeLabelScaler()

    raise NotImplementedError("Not supported")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input data *.tsv serializer')

    # Providing arguments.
    RuAttitudesVersionArg.add_argument(parser)
    LabelsCountArg.add_argument(parser)
    CvCountArg.add_argument(parser)
    ExperimentTypeArg.add_argument(parser)
    BertFormatterArg.add_argument(parser)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading arguments.
    exp_type = ExperimentTypeArg.read_argument(args)
    labels_count = LabelsCountArg.read_argument(args)
    cv_count = CvCountArg.read_argument(args)
    ra_version = RuAttitudesVersionArg.read_argument(args)
    bert_input_formatter = BertFormatterArg.read_argument(args)

    # Initialize logging.
    stream_handler = logging.StreamHandler()
    log_formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    stream_handler.setFormatter(log_formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    labels_scaler = create_labels_scaler(labels_count)
    data_io = RuSentRelBasedExperimentsIOUtils(labels_scaler=labels_scaler,
                                               init_word_embedding=False)

    if cv_count == 1:
        # Non cv mode.
        cv_mode = u''
        handler = __non_cv_experiment
    else:
        # Using cross validation.
        cv_mode = u'cv-'
        handler = __cv_based_experiment

    model_name = u"{cv_m}{training_type}-bert-{formatter}-{labels_mode}l".format(
        cv_m=cv_mode,
        training_type=exp_type,
        formatter=bert_input_formatter,
        labels_mode=int(labels_count))

    logger.info("Model name: {}".format(model_name))

    data_io.set_model_name(model_name)

    # Initialize experiment.
    experiment = None
    if exp_type == SUPERVISED_LEARNING_WITH_DS:
        ra = RuSentRelWithRuAttitudesExperiment.read_ruattitudes_in_memory(
            stemmer=data_io.Stemmer,
            version=ra_version)
        experiment = RuSentRelWithRuAttitudesExperiment(
            data_io=data_io,
            prepare_model_root=True,
            ra_instance=ra)
    elif exp_type == SUPERVISED_LEARNING:
        experiment = RuSentRelExperiment(data_io, True)

    # Running *.tsv serialization.
    handler(experiment=experiment,
            formatter=bert_input_formatter)
