#!/usr/bin/python
import argparse
import logging

from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler

from arekit.contrib.bert.core.input.io_utils import BertIOUtils
from arekit.contrib.bert.entity.str_rus_nocased_fmt import RussianEntitiesFormatter
from arekit.contrib.bert.run_serializer import BertExperimentInputSerializer
from arekit.contrib.bert.supported import SampleFormattersService
from arekit.contrib.experiments.rusentrel.experiment import RuSentRelExperiment
from arekit.contrib.experiments.rusentrel_ds.experiment import RuSentRelWithRuAttitudesExperiment
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions

from args.bert_formatter import BertFormatterArg
from args.cv_index import CvCountArg
from args.experiment import ExperimentTypeArg, SUPERVISED_LEARNING, SUPERVISED_LEARNING_WITH_DS
from args.labels_count import LabelsCountArg
from args.ra_ver import RuAttitudesVersionArg


# TODO. Move this into args.
from serializing_data import BertRuSentRelBasedSerializaingData


def create_labels_scaler(labels_count):
    assert (isinstance(labels_count, int))

    if labels_count == 2:
        return TwoLabelScaler()
    if labels_count == 3:
        return ThreeLabelScaler()

    raise NotImplementedError("Not supported")


def create_experiment(exp_type, data_io):
    if exp_type == SUPERVISED_LEARNING_WITH_DS:
        return RuSentRelWithRuAttitudesExperiment(data_io=data_io,
                                                  ruattitudes_version=ruattitudes_version,
                                                  rusentrel_version=rusentrel_version)

    elif exp_type == SUPERVISED_LEARNING:
        return RuSentRelExperiment(data_io=data_io, version=rusentrel_version)

    raise NotImplementedError("Experiment type {} is not supported!".format(exp_type))


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
    ruattitudes_version = RuAttitudesVersionArg.read_argument(args)
    sample_formatter_type = BertFormatterArg.read_argument(args)
    # TODO. Move this into input parameter.
    rusentrel_version = RuSentRelVersions.V11
    # TODO. Move this into input parameter.
    terms_per_context = 50
    # TODO. Move this into input parameter.
    entity_fmt = RussianEntitiesFormatter()

    # Initialize logging.
    stream_handler = logging.StreamHandler()
    log_formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    stream_handler.setFormatter(log_formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    label_scaler = create_labels_scaler(labels_count)
    serializing_data = BertRuSentRelBasedSerializaingData(labels_scaler=label_scaler,
                                                          terms_per_context=terms_per_context)

    # Setup model name.
    model_name = u"bert-{formatter}-{labels_mode}l".format(
        formatter=SampleFormattersService.type_to_value(sample_formatter_type),
        labels_mode=int(labels_count))
    logger.info("Model name: {}".format(model_name))
    serializing_data.set_model_name(model_name)

    # Initialize experiment.
    experiment = create_experiment(exp_type=exp_type,
                                   data_io=serializing_data)

    # Running *.tsv serialization.
    experiment.DataIO.CVFoldingAlgorithm.set_cv_count(cv_count)

    serializer = BertExperimentInputSerializer(experiment=experiment,
                                               skip_if_folder_exists=True,
                                               sample_formatter_type=sample_formatter_type,
                                               entity_formatter=entity_fmt,
                                               label_scaler=label_scaler,
                                               write_sample_header=True,
                                               io_utils=BertIOUtils)

    serializer.run()
