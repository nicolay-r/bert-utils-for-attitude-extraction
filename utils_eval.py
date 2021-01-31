from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.folding.types import FoldingType
from arekit.common.experiment.input.readers.opinion import InputOpinionReader
from arekit.common.experiment.output.multiple import MulticlassOutput
from arekit.common.experiment.output.opinions.converter import OutputToOpinionCollectionsConverter
from arekit.common.experiment.output.opinions.writer import save_opinion_collections
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.contrib.experiments.factory import create_experiment
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from experiment_io import CustomBertIOUtils


if __name__ == "__main__":

    # TODO. Provide these parameters.
    opinions_tsv_filepath = None
    result_filepath = None
    # From args.
    labels_scaler = None
    cmp_doc_ids_set = None
    exp_type = None
    experiment_data = None
    folding_type = None
    rusentrel_version = None
    load_ruattitude_docs = None
    experiment_io_type = None
    extra_name_suffix = None
    cv_count = None
    ra_version = None

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
    cmp_doc_ids_set = set(experiment.DocumentOperations.iter_doc_ids_to_compare())
    exp_io = experiment.ExperimentIO
    opin_fmt = experiment.DataIO.OpinionFormatter
    labels_formatter = RuSentRelLabelsFormatter()

    # iterate opinion collections.
    collections_iter = OutputToOpinionCollectionsConverter.iter_opinion_collections(
        output_filepath=result_filepath,
        opinions_reader=InputOpinionReader.from_tsv(opinions_tsv_filepath),
        labels_scaler=labels_scaler,
        create_opinion_collection_func=experiment.OpinionOperations.create_opinion_collection,
        keep_doc_id_func=lambda doc_id: doc_id in cmp_doc_ids_set,
        label_calculation_mode=LabelCalculationMode.AVERAGE,
        output=MulticlassOutput(labels_scaler),
        keep_news_ids_from_samples_reader=True,
        keep_ids_from_samples_reader=False)

    # Save collection.
    epoch_index = 0

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
