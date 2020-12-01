#!/usr/bin/python
import argparse
from os.path import exists
from tqdm import tqdm

from arekit.common.experiment.data.base import DataIO
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler
from arekit.common.experiment.data_type import DataType
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.contrib.bert.samplers.types import BertSampleFormatterTypes
from arekit.contrib.experiments.rusentrel.experiment import RuSentRelExperiment
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from experiment_data import CustomSerializationData


def perform_evaluation(cv_count, exp_data, formatter):
    assert(isinstance(formatter, unicode))
    assert(isinstance(exp_data, DataIO))

    model_name = u"{training_type}bert-{formatter}-{label_scale_type}l".format(
        training_type=u'cv-' if cv_count > 1 else u'',
        formatter=formatter,
        label_scale_type=str(exp_data.LabelsScaler))

    exp_data.set_model_name(model_name)
    exp_data.CVFoldingAlgorithm.set_cv_count(cv_count)

    if not exists(exp_data.get_model_results_root()):
        return

    experiment = RuSentRelExperiment(exp_data=exp_data,
                                     version=RuSentRelVersions.V11)

    eval_tsv(formatter_type=formatter,
             data_type=DataType.Test,
             experiment=experiment,
             label_calculation_mode=LabelCalculationMode.AVERAGE,
             labels_formatter=labels_formatter)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Bert results evaluation.')
    parser.add_argument('sources_dir',
                        metavar='src_dir',
                        type=str,
                        nargs=1,
                        help='Source of encode experiments (input, opinions for a particular cv)')

    parser.add_argument('results_dir',
                        metavar='res_dir',
                        type=str,
                        nargs=1,
                        help='Model results dir, with the corresponding to the source folder experiment results')

    args = parser.parse_args()

    formatters = [BertSampleFormatterTypes.CLASSIF_M,
                  BertSampleFormatterTypes.QA_M,
                  BertSampleFormatterTypes.NLI_M,
                  BertSampleFormatterTypes.QA_B,
                  BertSampleFormatterTypes.NLI_B]

    labels_formatter = ThreeScaleLabelsFormatter()

    # For cv_types.
    for cv_count in [1, 3]:

        # For label_scale.
        for label_scaler in [TwoLabelScaler(), ThreeLabelScaler()]:

            exp_data = CustomSerializationData(labels_scaler=label_scaler,
                                               init_word_embedding=False)

            exp_data.set_experiment_sources_dir(args.sources_dir[0].decode('utf-8'))
            exp_data.set_experiment_results_dir(args.results_dir[0].decode('utf-8'))

            print(u"cv{cv_count}-l{scaler}".format(cv_count=cv_count,
                                                   scaler=label_scaler))

            it_fmts = tqdm(desc=u"Performing evaluation...",
                           iterable=formatters,
                           unit=u"fmts",
                           ncols=120)

            # For each formatter.
            for formatter in it_fmts:
                perform_evaluation(cv_count=cv_count,
                                   formatter=formatter,
                                   exp_data=exp_data)
