#!/usr/bin/python
import argparse
import logging

from arekit.common.entities.formatters.factory import create_entity_formatter
from arekit.common.experiment.folding.types import FoldingType
from arekit.common.experiment.scales.factory import create_labels_scaler

from arekit.contrib.bert.run_serializer import BertExperimentInputSerializer
from arekit.contrib.experiments.factory import create_experiment

from arekit.processing.pos.mystem_wrap import POSMystemWrapper
from args.balance import UseBalancingArg
from args.bert_formatter import BertInputFormatterArg

from args.cv_index import CvCountArg
from args.dist_in_terms_between_ends import DistanceInTermsBetweenAttitudeEndsArg
from args.entity_fmt import EnitityFormatterTypesArg
from args.experiment import ExperimentTypeArg
from args.frames import RuSentiFramesVersionArg
from args.labels_count import LabelsCountArg
from args.ra_ver import RuAttitudesVersionArg
from args.rusentrel import RuSentRelVersionArg
from args.stemmer import StemmerArg
from args.terms_per_context import TermsPerContextArg
from bert_model_io import BertModelIO
from common import create_full_model_name
from experiment_data import CustomSerializationData
from experiment_io import CustomBertIOUtils


def create_exp_name_suffix(use_balancing, terms_per_context, dist_in_terms_between_att_ends):
    """ Provides an external parameters that assumes to be synchronized both
        by serialization and training experiment stages.
    """
    assert (isinstance(use_balancing, bool))
    assert (isinstance(terms_per_context, int))
    assert (isinstance(dist_in_terms_between_att_ends, int) or dist_in_terms_between_att_ends is None)

    # You may provide your own parameters out there
    params = [
        u"balanced" if use_balancing else u"nobalance",
        u"tpc{}".format(terms_per_context)
    ]

    if dist_in_terms_between_att_ends is not None:
        params.append(u"dbe{}".format(dist_in_terms_between_att_ends))

    return u'-'.join(params)


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
    BertInputFormatterArg.add_argument(parser)
    EnitityFormatterTypesArg.add_argument(parser)
    StemmerArg.add_argument(parser)
    DistanceInTermsBetweenAttitudeEndsArg.add_argument(parser)
    UseBalancingArg.add_argument(parser)

    parser.add_argument('--parse-frames',
                        dest='parse_frames',
                        type=bool,
                        const=True,
                        default=False,
                        nargs='?',
                        help='Perform frames parsing')

    parser.add_argument('--no-balancing',
                        dest='balancing_disabled',
                        type=bool,
                        const=True,
                        default=False,
                        nargs='?',
                        help='Disable balancing for Train type during sample serialization process')

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
    balance_samples = UseBalancingArg.read_argument(args)
    sample_formatter_type = BertInputFormatterArg.read_argument(args)
    entity_formatter_type = EnitityFormatterTypesArg.read_argument(args)
    dist_in_terms_between_attitude_ends = DistanceInTermsBetweenAttitudeEndsArg.read_argument(args)
    stemmer = StemmerArg.read_argument(args)
    parse_frames = args.parse_frames

    # Initialize logging.
    stream_handler = logging.StreamHandler()
    log_formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    stream_handler.setFormatter(log_formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # Initialize entity formatter.
    entity_formatter = create_entity_formatter(
        fmt_type=entity_formatter_type,
        create_russian_pos_tagger_func=lambda: POSMystemWrapper(mystem=stemmer.MystemInstance))

    # Setup model name.
    full_model_name = create_full_model_name(sample_fmt_type=sample_formatter_type,
                                             entities_fmt_type=entity_formatter_type,
                                             labels_count=labels_count)

    model_io = BertModelIO(full_model_name=full_model_name)

    # Create experiment data and all the related information.
    experiment_data = CustomSerializationData(
        labels_scaler=create_labels_scaler(labels_count),
        stemmer=stemmer,
        frames_version=frames_version if parse_frames else None,
        model_io=model_io,
        terms_per_context=terms_per_context)

    extra_name_suffix = create_exp_name_suffix(
        use_balancing=balance_samples,
        terms_per_context=terms_per_context,
        dist_in_terms_between_att_ends=dist_in_terms_between_attitude_ends)

    # Initialize experiment.
    experiment = create_experiment(exp_type=exp_type,
                                   experiment_data=experiment_data,
                                   folding_type=FoldingType.Fixed if cv_count == 1 else FoldingType.CrossValidation,
                                   rusentrel_version=rusentrel_version,
                                   experiment_io_type=CustomBertIOUtils,
                                   ruattitudes_version=ra_version,
                                   load_ruattitude_docs=True,
                                   extra_name_suffix=extra_name_suffix)

    engine = BertExperimentInputSerializer(experiment=experiment,
                                           skip_if_folder_exists=False,
                                           sample_formatter_type=sample_formatter_type,
                                           entity_formatter=entity_formatter,
                                           balance_train_samples=balance_samples)

    engine.run()
