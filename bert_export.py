import logging
import sys

sys.path.append('../')

from arekit.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.common.experiment.scales.two import TwoLabelScaler
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.contrib.experiments.rusentrel.experiment import RuSentRelExperiment
from arekit.contrib.experiments.rusentrel_ds.experiment import RuSentRelWithRuAttitudesExperiment
from arekit.contrib.bert.formatters.sample.formats import SampleFormatters
from arekit.contrib.bert.encoder import BertEncoder
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


if __name__ == "__main__":

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    ##########
    # Compose
    ##########
    data_io_modes = [
        (u"3", RuSentRelBasedExperimentsIOUtils(labels_scaler=ThreeLabelScaler(), init_word_embedding=False)),
        (u"2", RuSentRelBasedExperimentsIOUtils(labels_scaler=TwoLabelScaler(), init_word_embedding=False))]

    cv_modes = {
        # u'': __non_cv_experiment,
        u'cv-': __cv_based_experiment
    }

    DS_TRAINING_TYPE = u'ds-'

    training_modes = [(DS_TRAINING_TYPE, RuSentRelWithRuAttitudesExperiment),
                      (u'', RuSentRelExperiment)]

    stemmer = data_io_modes[0][1].Stemmer
    logger.info("Preparing ...")

    # TODO. This might be (version) modified in order to calculate data for other collections.
    ra_v11 = RuSentRelWithRuAttitudesExperiment.read_ruattitudes_in_memory(
        stemmer=stemmer,
        version=RuAttitudesVersions.V11)

    for labels_mode, data_io in data_io_modes:
        for training_type, experiment_type in training_modes:
            for formatter in SampleFormatters.iter_supported():

                for cv_mode, handler in cv_modes.iteritems():

                    model_name = u"{cv_m}{training_type}bert-{formatter}-{labels_mode}l".format(
                        cv_m=cv_mode,
                        training_type=training_type,
                        formatter=formatter,
                        labels_mode=labels_mode)

                    logger.info("Model name: {}".format(model_name))

                    data_io.set_model_name(model_name)

                    if training_type == DS_TRAINING_TYPE:
                        experiment = RuSentRelWithRuAttitudesExperiment(
                            data_io=data_io,
                            prepare_model_root=True,
                            ra_instance=ra_v11)
                    else:
                        experiment = RuSentRelExperiment(data_io, True)

                    handler(experiment=experiment, formatter=formatter)
