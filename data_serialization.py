from arekit.contrib.experiment_rusentrel.annot.algo import RuSentRelDefaultNeutralAnnotationAlgorithm
from arekit.contrib.experiment_rusentrel.annot.factory import ExperimentAnnotatorFactory
from arekit.contrib.experiment_rusentrel.frame_variants import ExperimentFrameVariantsCollection
from arekit.contrib.experiment_rusentrel.labels.formatters.rusentiframes import ExperimentRuSentiFramesLabelsFormatter, \
    ExperimentRuSentiFramesEffectLabelsFormatter

from arekit.common.experiment.data.serializing import SerializationData

from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter

from bert_model_io import BertModelIO


class CustomSerializationData(SerializationData):

    def __init__(self, labels_scaler, stemmer, frames_version, model_io, terms_per_context,
                 dist_in_terms_between_attitude_ends):
        assert(isinstance(frames_version, RuSentiFramesVersions) or frames_version is None)
        assert(isinstance(model_io, BertModelIO))
        assert(isinstance(terms_per_context, int))

        self.__dist_in_terms_between_attitude_ends = dist_in_terms_between_attitude_ends

        annot = ExperimentAnnotatorFactory.create(
            labels_count=labels_scaler.LabelsCount,
            create_algo=lambda: RuSentRelDefaultNeutralAnnotationAlgorithm(
                dist_in_terms_bound=dist_in_terms_between_attitude_ends))

        super(CustomSerializationData, self).__init__(
            label_scaler=labels_scaler, annot=annot, stemmer=stemmer)

        self.__model_io = model_io
        self.__terms_per_context = terms_per_context
        self.__opinion_formatter = RuSentRelOpinionCollectionFormatter()

        self.__frames_collection = None
        self.__unique_frame_variants = None

        if frames_version is not None:

            self.__frames_collection = RuSentiFramesCollection.read_collection(
                version=frames_version,
                labels_fmt=ExperimentRuSentiFramesLabelsFormatter(),
                effect_labels_fmt=ExperimentRuSentiFramesEffectLabelsFormatter())

            self.__unique_frame_variants = ExperimentFrameVariantsCollection(stemmer)

            # Filling collection.
            self.__unique_frame_variants.fill_from_iterable(
                variants_with_id=self.__frames_collection.iter_frame_id_and_variants())

    # region public properties

    @property
    def ModelIO(self):
        return self.__model_io

    @property
    def FramesCollection(self):
        return self.__frames_collection

    @property
    def FrameVariantCollection(self):
        return self.__unique_frame_variants

    @property
    def OpinionFormatter(self):
        return self.__opinion_formatter

    @property
    def DistanceInTermsBetweenOpinionEndsBound(self):
        return self.__dist_in_terms_between_attitude_ends

    @property
    def TermsPerContext(self):
        return self.__terms_per_context

    # endregion
