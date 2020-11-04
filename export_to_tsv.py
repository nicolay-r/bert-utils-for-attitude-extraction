#!/usr/bin/python
import argparse
import logging

from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.experiment.data.serializing import SerializationData
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.encoder import BaseInputEncoder
from arekit.common.experiment.input.formatters.opinion import BaseOpinionsFormatter
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler

from arekit.contrib.bert.core.input.io_utils import BertIOUtils
from arekit.contrib.bert.factory import create_bert_sample_formatter
from arekit.contrib.bert.supported import SampleFormattersService
from arekit.contrib.experiments.rusentrel.experiment import RuSentRelExperiment
from arekit.contrib.experiments.rusentrel_ds.experiment import RuSentRelWithRuAttitudesExperiment

from args.cv_index import CvCountArg
from args.entity_fmt import EnitityFormatterTypesArg
from args.experiment import ExperimentTypeArg, SUPERVISED_LEARNING, SUPERVISED_LEARNING_WITH_DS
from args.frames import RuSentiFramesVersionArg
from args.labels_count import LabelsCountArg
from args.ra_ver import RuAttitudesVersionArg
from args.rusentrel import RuSentRelVersionArg
from args.stemmer import StemmerArg
from args.terms_per_context import TermsPerContextArg


def create_labels_scaler(labels_count):
    assert (isinstance(labels_count, int))

    if labels_count == 2:
        return TwoLabelScaler()
    if labels_count == 3:
        return ThreeLabelScaler()

    raise NotImplementedError("Not supported")


def create_experiment(exp_type, data_io):
    prepare_model_root = False

    if exp_type == SUPERVISED_LEARNING_WITH_DS:
        return RuSentRelWithRuAttitudesExperiment(data_io=data_io,
                                                  prepare_model_root=prepare_model_root,
                                                  ruattitudes_version=ruattitudes_version,
                                                  rusentrel_version=rusentrel_version)

    elif exp_type == SUPERVISED_LEARNING:
        return RuSentRelExperiment(data_io=data_io,
                                   version=rusentrel_version,
                                   prepare_model_root=prepare_model_root)

    raise NotImplementedError("Experiment type {} is not supported!".format(exp_type))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input data *.tsv serializer')

    # Providing arguments.
    RuAttitudesVersionArg.add_argument(parser)
    CvCountArg.add_argument(parser)
    ExperimentTypeArg.add_argument(parser)
    LabelsCountArg.add_argument(parser)
    TermsPerContextArg.add_argument(parser)
    RuSentiFramesVersionArg.add_argument(parser)
    RuSentRelVersionArg.add_argument(parser)
    EnitityFormatterTypesArg.add_argument(parser)
    StemmerArg.add_argument(parser)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading arguments.
    exp_type = ExperimentTypeArg.read_argument(args)
    ra_version = RuAttitudesVersionArg.read_argument(args)
    cv_count = CvCountArg.read_argument(args)
    labels_count = LabelsCountArg.read_argument(args)
    terms_per_context = TermsPerContextArg.read_argument(args)
    frames_version = RuSentiFramesVersionArg.read_argument(args)
    rusentrel_version = RuSentRelVersionArg.read_argument(args)
    entity_fmt = EnitityFormatterTypesArg.read_argument(args)
    stemmer = StemmerArg.read_argument(args)

    # Initialize logging.
    stream_handler = logging.StreamHandler()
    log_formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    stream_handler.setFormatter(log_formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    label_scaler = create_labels_scaler(labels_count)
    experiment_data = SerializationData(labels_scaler=label_scaler,
                                        stemmer=stemmer)

    # Setup model name.
    model_name = u"bert-{formatter}-{labels_mode}l".format(
        formatter=SampleFormattersService.type_to_value(sample_formatter_type),
        labels_mode=int(labels_count))
    logger.info("Model name: {}".format(model_name))
    experiment_data.set_model_name(model_name)

    # Initialize experiment.
    experiment = create_experiment(exp_type=exp_type,
                                   data_io=experiment_data)

    # Running *.tsv serialization.
    experiment.DataIO.CVFoldingAlgorithm.set_cv_count(cv_count)

    # Create data type.
    data_type = DataType.Train

    # Create samples formatter.
    sample_formatter = create_bert_sample_formatter(data_type=data_type,
                                                    formatter_type=sample_fmt,
                                                    label_scaler=label_scaler,
                                                    entity_formatter=entity_fmt)

    # Load parsed news collections in memory.
    # Taken from Neural networks formatter.
    parsed_news_collection = experiment.create_parsed_collection(data_type)

    # Compose text opinion helper.
    # Taken from Neural networks formatter.
    text_opinion_helper = TextOpinionHelper(lambda news_id: parsed_news_collection.get_by_news_id(news_id))

    # TODO. Call this for multiple data_types
    BaseInputEncoder.to_tsv(
        sample_filepath=BertIOUtils.get_input_sample_filepath(data_type=data_type),
        opinion_filepath=BertIOUtils.get_input_opinions_filepath(data_type=data_type),
        opinion_formatter=BaseOpinionsFormatter(data_type),
        opinion_provider=OpinionProvider.from_experiment(
            experiment=experiment,
            data_type=data_type,
            iter_news_ids=parsed_news_collection.iter_news_ids(),
            terms_per_context=terms_per_context,
            text_opinion_helper=text_opinion_helper),
        sample_formatter=sample_formatter,
        # TODO. Move this into arguments.
        write_sample_header=True)
