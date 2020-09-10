#!/usr/bin/python
import argparse
import sys
from os.path import exists
from tqdm import tqdm

sys.path.append('..')

from arekit.common.experiment.data_io import DataIO
from arekit.common.experiment.neutral.annot.labels_fmt import ThreeScaleLabelsFormatter
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler
from arekit.common.experiment.data_type import DataType
from arekit.common.model.labeling.modes import LabelCalculationMode

from arekit.contrib.bert.eval_tsv import eval_tsv
from arekit.contrib.experiments.rusentrel.experiment import RuSentRelExperiment
from arekit.contrib.bert.formatters.sample.formats import SampleFormatters

from io_utils import RuSentRelBasedExperimentsIOUtils


def perform_evaluation(cv_count, data_io, formatter):
    assert(isinstance(formatter, unicode))
    assert(isinstance(data_io, DataIO))

    model_name = u"{training_type}bert-{formatter}-{label_scale_type}l".format(
        training_type=u'cv-' if cv_count > 1 else u'',
        formatter=formatter,
        label_scale_type=str(data_io.LabelsScaler))

    data_io.set_model_name(model_name)
    data_io.CVFoldingAlgorithm.set_cv_count(cv_count)

    if not exists(data_io.get_model_results_root()):
        return

    experiment = RuSentRelExperiment(data_io=data_io,
                                     prepare_model_root=False)

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

    formatters = [SampleFormatters.CLASSIF_M,
                  SampleFormatters.QA_M,
                  SampleFormatters.NLI_M,
                  SampleFormatters.QA_B,
                  SampleFormatters.NLI_B]

    labels_formatter = ThreeScaleLabelsFormatter()

    # For cv_types.
    for cv_count in [1, 3]:

        # For label_scale.
        for label_scaler in [TwoLabelScaler(), ThreeLabelScaler()]:

            data_io = RuSentRelBasedExperimentsIOUtils(labels_scaler=label_scaler,
                                                       init_word_embedding=False)

            data_io.set_experiment_sources_dir(args.sources_dir[0].decode('utf-8'))
            data_io.set_experiment_results_dir(args.results_dir[0].decode('utf-8'))

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
                                   data_io=data_io)

