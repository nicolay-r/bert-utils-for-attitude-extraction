from bert_model_io import BertModelIO

from arekit.common.experiment.data.serializing import SerializationData
from arekit.common.frame_variants.collection import FrameVariantsCollection

from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter


class CustomSerializationData(SerializationData):

    def __init__(self, labels_scaler, stemmer, frames_version, model_io, terms_per_context):
        assert(isinstance(frames_version, RuSentiFramesVersions) or frames_version is None)
        assert(isinstance(model_io, BertModelIO))
        assert(isinstance(terms_per_context, int))

        super(CustomSerializationData, self).__init__(labels_scaler, stemmer)

        self.__model_io = model_io
        self.__terms_per_context = terms_per_context
        self.__opinion_formatter = RuSentRelOpinionCollectionFormatter()

        self.__frames_collection = None
        self.__unique_frame_variants = None

        if frames_version is not None:
            self.__frames_collection = RuSentiFramesCollection.read_collection(version=frames_version)
            self.__unique_frame_variants = FrameVariantsCollection.create_unique_variants_from_iterable(
                variants_with_id=self.__frames_collection.iter_frame_id_and_variants(),
                stemmer=stemmer)

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
        return 10

    @property
    def TermsPerContext(self):
        return self.__terms_per_context

    # endregion
