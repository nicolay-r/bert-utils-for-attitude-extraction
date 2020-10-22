#!/usr/bin/python
import argparse
import logging

from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.encoder import BaseInputEncoder
from arekit.common.experiment.input.formatters.opinion import BaseOpinionsFormatter
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler
from arekit.contrib.bert.entity.str_rus_nocased_fmt import RussianEntitiesFormatter
from arekit.contrib.bert.factory import create_bert_sample_formatter
from arekit.contrib.bert.supported import SampleFormattersService
from arekit.contrib.experiments.ruattitudes.utils import read_ruattitudes_in_memory
from arekit.contrib.experiments.rusentrel.experiment import RuSentRelExperiment
from arekit.contrib.experiments.rusentrel_ds.experiment import RuSentRelWithRuAttitudesExperiment
from arekit.contrib.networks.core.io_utils import NetworkIOUtils
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions

from args.bert_formatter import BertFormatterArg
from args.cv_index import CvCountArg
from args.experiment import ExperimentTypeArg, SUPERVISED_LEARNING, SUPERVISED_LEARNING_WITH_DS
from args.labels_count import LabelsCountArg
from args.ra_ver import RuAttitudesVersionArg

from io_utils import RuSentRelBasedExperimentsIOUtils


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
    rusentrel_version = RuSentRelVersions.V11
    sample_formatter_type = BertFormatterArg.read_argument(args)
    terms_per_context = 50
    entity_fmt = RussianEntitiesFormatter()

    # Initialize logging.
    stream_handler = logging.StreamHandler()
    log_formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    stream_handler.setFormatter(log_formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    label_scaler = create_labels_scaler(labels_count)
    data_io = RuSentRelBasedExperimentsIOUtils(labels_scaler=label_scaler,
                                               init_word_embedding=False)

    cv_mode = u'' if cv_count == 1 else u'cv-'

    model_name = u"{cv_m}{training_type}-bert-{formatter}-{labels_mode}l".format(
        cv_m=cv_mode,
        training_type=exp_type,
        formatter= SampleFormattersService.type_to_value(sample_formatter_type),
        labels_mode=int(labels_count))

    logger.info("Model name: {}".format(model_name))
    data_io.set_model_name(model_name)

    # Initialize experiment.
    experiment = None
    if exp_type == SUPERVISED_LEARNING_WITH_DS:
        ra = read_ruattitudes_in_memory(version=ra_version)
        experiment = RuSentRelWithRuAttitudesExperiment(
            data_io=data_io,
            prepare_model_root=True,
            ruattitudes_version=ra_version,
            rusentrel_version=rusentrel_version,
            ra_instance=ra)

    elif exp_type == SUPERVISED_LEARNING:
        experiment = RuSentRelExperiment(data_io=data_io,
                                         version=rusentrel_version,
                                         prepare_model_root=False)

    # Running *.tsv serialization.
    experiment.DataIO.CVFoldingAlgorithm.set_cv_count(cv_count)

    # Create data type.
    data_type = DataType.Train

    # Create samples formatter.
    sample_formatter = create_bert_sample_formatter(data_type=data_type,
                                                    formatter_type=sample_formatter_type,
                                                    label_scaler=label_scaler,
                                                    entity_formatter=entity_fmt)

    # Load parsed news collections in memory.
    # Taken from Neural networks formatter.
    parsed_news_collection = experiment.create_parsed_collection(data_type)

    # Compose text opinion helper.
    # Taken from Neural networks formatter.
    text_opinion_helper = TextOpinionHelper(lambda news_id: parsed_news_collection.get_by_news_id(news_id))

    BaseInputEncoder.to_tsv(
        sample_filepath=NetworkIOUtils.get_input_sample_filepath(experiment=experiment, data_type=data_type),
        opinion_filepath=NetworkIOUtils.get_input_opinions_filepath(experiment=experiment, data_type=data_type),
        opinion_formatter=BaseOpinionsFormatter(data_type),
        opinion_provider=OpinionProvider.from_experiment(
            experiment=experiment,
            data_type=data_type,
            iter_news_ids=parsed_news_collection.iter_news_ids(),
            terms_per_context=terms_per_context,
            text_opinion_helper=text_opinion_helper),
        sample_formatter=sample_formatter,
        write_sample_header=True)
