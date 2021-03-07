import argparse
from arekit.common.entities.formatters.types import EntityFormattersService
from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.folding.types import FoldingType
from arekit.common.experiment.scales.factory import create_labels_scaler

from arekit.contrib.bert.output.eval_helper import EvalHelper
from arekit.contrib.bert.run_evaluation import LanguageModelExperimentEvaluator
from arekit.contrib.bert.samplers.types import BertSampleFormatterTypes
from arekit.contrib.experiments.factory import create_experiment
from arekit.contrib.experiments.types import ExperimentTypes
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersionsService
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersionsService
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter

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

    def __init__(self, log_dir, state_name, ft_tag):
        assert(isinstance(ft_tag, unicode) or ft_tag is None)
        self.__log_dir = log_dir
        self.__state_name = state_name
        self.__ft_tag = ft_tag

    def __create_results_filename(self, iter_index, epoch_index):
        return CustomEvalHelper.RESULTS_TEMPLATE_FILENAME.format(it_index=iter_index,
                                                                 epoch_index=epoch_index,
                                                                 state_name=self.__state_name)

    def __get_results_dir(self, target_dir):
        return Common.combine_tag_with_full_model_name(full_model_name=target_dir,
                                                       tag=self.__ft_tag)

    def get_results_dir(self, target_dir):
        return self.__get_results_dir(target_dir)

    def get_results_filename(self, iter_index, epoch_index):
        return self.__create_results_filename(iter_index=iter_index, epoch_index=epoch_index)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='*.tsv results based evaluator')

    parser.add_argument('--max-epochs',
                        dest="max_epochs",
                        type=int,
                        default=200,
                        nargs='?',
                        help="Labels count in an output classifier")

    # Parsing arguments.
    args = parser.parse_args()

    # Constant predefined parameters.
    max_epochs_count = args.max_epochs
    rusentrel_version = RuSentRelVersionArg.default
    terms_per_context = TermsPerContextArg.default
    stemmer = StemmerArg.supported[StemmerArg.default]
    eval_mode = EvaluationModes.Extraction
    dist_in_terms_between_attitude_ends = None

    # grid for looking through.
    grid = {
        u"labels": [2, 3],
        u"foldings": [FoldingType.Fixed,
                      FoldingType.CrossValidation],
        u"exp_types": [ExperimentTypes.RuSentRel,
                       ExperimentTypes.RuSentRelWithRuAttitudes],
        u"entity_fmts": [EntityFormattersService.get_type_by_name(ent_fmt)
                         for ent_fmt in EntityFormattersService.iter_supported_names()],
        u"sample_types": [fmt_type for fmt_type in BertSampleFormatterTypes],
        u"ra_names": [RuAttitudesVersionsService.find_by_name(ra_name)
                      for ra_name in RuAttitudesVersionsService.iter_supported_names()],
        u'balancing': [True],
        u"frames_versions": [RuSentiFramesVersionsService.get_type_by_name(frames_version)
                             for frames_version in RuSentiFramesVersionsService.iter_supported_names()],
        u"state_names": [u"ra-12-bert-base-nli-pretrained-2l",
                         u"multi_cased_L-12_H-768_A-12"]
    }

    def __run():

        full_model_name = Common.create_full_model_name(
            sample_fmt_type=sample_formatter_type,
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
                                       folding_type=folding_type,
                                       rusentrel_version=rusentrel_version,
                                       experiment_io_type=CustomBertIOUtils,
                                       ruattitudes_version=ra_version,
                                       load_ruattitude_docs=False,
                                       extra_name_suffix=extra_name_suffix)

        eval_helper = CustomEvalHelper(log_dir=Common.log_dir,
                                       state_name=state_name,
                                       ft_tag=Common.get_tag_by_ruattitudes_version(ra_version))

        engine = LanguageModelExperimentEvaluator(experiment=experiment,
                                                  data_type=DataType.Test,
                                                  eval_helper=eval_helper,
                                                  max_epochs_count=max_epochs_count)

        # Starting evaluation process.
        engine.run()

    for labels_count in grid[u"labels"]:
        for folding_type in grid[u"foldings"]:
            for exp_type in grid[u'exp_types']:
                for entity_formatter_type in grid[u'entity_fmts']:
                    for sample_formatter_type in grid[u'sample_types']:
                        for balance_samples in grid[u'balancing']:
                            for ra_version in grid[u'ra_names']:
                                for frames_version in grid[u'frames_versions']:
                                    for state_name in grid[u'state_names']:
                                        __run()
