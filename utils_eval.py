import argparse
from os.path import exists, join

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.folding.types import FoldingType
from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.experiment.input.readers.opinion import InputOpinionReader
from arekit.common.experiment.input.readers.sample import InputSampleReader
from arekit.common.experiment.output.multiple import MulticlassOutput
from arekit.common.experiment.output.opinions.converter import OutputToOpinionCollectionsConverter
from arekit.common.experiment.output.opinions.writer import save_opinion_collections
from arekit.common.experiment.scales.factory import create_labels_scaler
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.contrib.experiments.factory import create_experiment
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
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
from run_serialization import create_exp_name_suffix

RESULTS_TEMPLATE_FILENAME = u"test_results_i{it_index}_e{epoch_index}_s{state_name}.tsv"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='*.tsv results based evaluator')

    ExperimentTypeArg.add_argument(parser)
    CvCountArg.add_argument(parser)
    RuSentRelVersionArg.add_argument(parser)
    LabelsCountArg.add_argument(parser)
    RuAttitudesVersionArg.add_argument(parser)
    UseBalancingArg.add_argument(parser)
    TermsPerContextArg.add_argument(parser)
    DistanceInTermsBetweenAttitudeEndsArg.add_argument(parser)
    StemmerArg.add_argument(parser)
    RuSentiFramesVersionArg.add_argument(parser)
    BertInputFormatterArg.add_argument(parser)
    EnitityFormatterTypesArg.add_argument(parser)

    # Parsing arguments.
    args = parser.parse_args()

    stemmer = StemmerArg.read_argument(args)
    labels_count = LabelsCountArg.read_argument(args)
    exp_type = ExperimentTypeArg.read_argument(args)
    cv_count = CvCountArg.read_argument(args)
    rusentrel_version = RuSentRelVersionArg.read_argument(args)
    sample_formatter_type = BertInputFormatterArg.read_argument(args)
    entity_formatter_type = EnitityFormatterTypesArg.read_argument(args)
    labels_scaler = create_labels_scaler(labels_count)
    ra_version = RuAttitudesVersionArg.read_argument(args)
    folding_type = FoldingType.Fixed if cv_count == 1 else FoldingType.CrossValidation
    balance_samples = UseBalancingArg.read_argument(args)
    terms_per_context = TermsPerContextArg.read_argument(args)
    dist_in_terms_between_attitude_ends = DistanceInTermsBetweenAttitudeEndsArg.read_argument(args)
    frames_version = RuSentiFramesVersionArg.read_argument(args)
    experiment_io_type = CustomBertIOUtils

    full_model_name = create_full_model_name(sample_fmt_type=sample_formatter_type,
                                             entities_fmt_type=entity_formatter_type,
                                             labels_count=int(labels_count))

    extra_name_suffix = create_exp_name_suffix(use_balancing=balance_samples,
                                               terms_per_context=terms_per_context,
                                               dist_in_terms_between_att_ends=dist_in_terms_between_attitude_ends)

    model_io = BertModelIO(full_model_name=full_model_name)
    experiment_data = CustomSerializationData(
        labels_scaler=create_labels_scaler(labels_count),
        stemmer=stemmer,
        frames_version=None,
        model_io=model_io,
        terms_per_context=terms_per_context)

    # Composing experiment.
    experiment = create_experiment(exp_type=exp_type,
                                   experiment_data=experiment_data,
                                   folding_type=FoldingType.Fixed if cv_count == 1 else FoldingType.CrossValidation,
                                   rusentrel_version=rusentrel_version,
                                   experiment_io_type=CustomBertIOUtils,
                                   ruattitudes_version=ra_version,
                                   load_ruattitude_docs=False,
                                   extra_name_suffix=extra_name_suffix)

    # Parameters required for evaluation.
    data_type = DataType.Test
    cmp_doc_ids_set = set(experiment.DocumentOperations.iter_doc_ids_to_compare())
    exp_io = experiment.ExperimentIO
    opin_fmt = experiment.DataIO.OpinionFormatter
    labels_formatter = RuSentRelLabelsFormatter()

    # TODO. bring epochs_count onto cmd_args level.
    for it_index in range(cv_count):

        # Providing opinions reader.
        opinions_tsv_filepath = exp_io.get_input_opinions_filepath(data_type=data_type)
        # Providing samples reader.
        samples_tsv_filepath = experiment.ExperimentIO.get_input_sample_filepath(data_type=data_type)

        for epoch_index in range(100):

            result_filename_template = RESULTS_TEMPLATE_FILENAME.format(
                it_index=it_index,
                epoch_index=epoch_index,
                # TODO. Bring this onto cmd_args level.
                state_name=u"multi_cased_L-12_H-768_A-12")

            result_filepath = join(experiment.ExperimentIO.get_target_dir(), result_filename_template)

            if not exists(result_filepath):
                continue

            print "Found:", result_filepath

            # iterate opinion collections.
            collections_iter = OutputToOpinionCollectionsConverter.iter_opinion_collections(
                output_filepath=result_filepath,
                read_header_in_output=False,
                opinions_reader=InputOpinionReader.from_tsv(opinions_tsv_filepath, compression='infer'),
                labels_scaler=labels_scaler,
                create_opinion_collection_func=experiment.OpinionOperations.create_opinion_collection,
                keep_doc_id_func=lambda doc_id: doc_id in cmp_doc_ids_set,
                label_calculation_mode=LabelCalculationMode.AVERAGE,
                output=MulticlassOutput(labels_scaler),
                samples_reader=InputSampleReader.from_tsv(filepath=samples_tsv_filepath,
                                                          row_ids_provider=MultipleIDProvider()))

            save_opinion_collections(
                opinion_collection_iter=collections_iter,
                create_file_func=lambda doc_id: exp_io.create_result_opinion_collection_filepath(
                    data_type=DataType.Test,
                    doc_id=doc_id,
                    epoch_index=epoch_index),
                save_to_file_func=lambda filepath, collection: opin_fmt.save_to_file(
                    collection=collection,
                    filepath=filepath,
                    labels_formatter=labels_formatter))

            # evaluate
            experiment.evaluate(data_type=DataType.Test,
                                epoch_index=epoch_index)
