import argparse

from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.folding.types import FoldingType
from arekit.common.experiment.scales.factory import create_labels_scaler
from arekit.contrib.bert.output.eval_helper import EvalHelper
from arekit.contrib.bert.run_evaluation import LanguageModelExperimentEvaluator
from arekit.contrib.experiments.factory import create_experiment
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
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
from callback import CustomCallback
from common import Common

from data_training import CustomTrainingData
from experiment_io import CustomBertIOUtils
from run_serialization import create_exp_name_suffix


class CustomEvalHelper(EvalHelper):

    RESULTS_TEMPLATE_FILENAME = u"test_results_i{it_index}_e{epoch_index}_s{state_name}.tsv"

    def __init__(self, log_dir, state_name):
        self.__log_dir = log_dir
        self.__state_name = state_name

    def get_results_filename(self, iter_index, epoch_index):
        return CustomEvalHelper.RESULTS_TEMPLATE_FILENAME.format(it_index=iter_index,
                                                                 epoch_index=epoch_index,
                                                                 state_name=self.__state_name)


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
    eval_mode = EvaluationModes.Extraction if labels_count == 3 else EvaluationModes.Classification

    full_model_name = Common.create_full_model_name(sample_fmt_type=sample_formatter_type,
                                                    entities_fmt_type=entity_formatter_type,
                                                    labels_count=int(labels_count))

    extra_name_suffix = create_exp_name_suffix(use_balancing=balance_samples,
                                               terms_per_context=terms_per_context,
                                               dist_in_terms_between_att_ends=dist_in_terms_between_attitude_ends)

    model_io = BertModelIO(full_model_name=full_model_name)

    # Setup default evaluator.
    evaluator = TwoClassEvaluator(eval_mode)

    experiment_data = CustomTrainingData(
        labels_scaler=create_labels_scaler(labels_count),
        stemmer=stemmer,
        evaluator=evaluator,
        opinion_formatter=RuSentRelOpinionCollectionFormatter(),
        model_io=model_io,
        callback=CustomCallback(DataType.Test))

    # Composing experiment.
    experiment = create_experiment(exp_type=exp_type,
                                   experiment_data=experiment_data,
                                   folding_type=FoldingType.Fixed if cv_count == 1 else FoldingType.CrossValidation,
                                   rusentrel_version=rusentrel_version,
                                   experiment_io_type=CustomBertIOUtils,
                                   ruattitudes_version=ra_version,
                                   load_ruattitude_docs=False,
                                   extra_name_suffix=extra_name_suffix)

    eval_helper = CustomEvalHelper(log_dir=Common.log_dir,
                                   state_name=u"multi_cased_L-12_H-768_A-12")

    engine = LanguageModelExperimentEvaluator(experiment=experiment,
                                              data_type=DataType.Test,
                                              eval_helper=eval_helper,
                                              max_epochs_count=100)

    # Starting evaluation process.
    engine.run()
