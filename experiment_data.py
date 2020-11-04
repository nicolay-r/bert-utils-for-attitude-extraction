import logging

from arekit.common.experiment.data.serializing import SerializationData
from arekit.common.frame_variants.collection import FrameVariantsCollection

from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.io_utils import RuSentiFramesVersions
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollection

from arekit.processing.lemmatization.mystem import MystemWrapper


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CustomSerializationData(SerializationData):

    def __init__(self, labels_scaler):
        self.__stemmer = MystemWrapper()
        super(CustomSerializationData, self).__init__(labels_scaler=labels_scaler,
                                                      stemmer=self.__stemmer)

        # TODO. Provide this from the outside
        self.__synonym_collection = RuSentRelSynonymsCollection.load_collection(
            stemmer=self.__stemmer,
            is_read_only=True)
        self.__opinion_formatter = RuSentRelOpinionCollectionFormatter(self.__synonym_collection)

        # TODO. Provide this from the outside
        self.__frames_collection = RuSentiFramesCollection.read_collection(version=RuSentiFramesVersions.V10)
        self.__unique_frame_variants = FrameVariantsCollection.create_unique_variants_from_iterable(
            variants_with_id=self.__frames_collection.iter_frame_id_and_variants(),
            stemmer=self.__stemmer)

    # region public properties

    @property
    def FramesCollection(self):
        return self.__frames_collection

    @property
    def FrameVariantCollection(self):
        return self.__unique_frame_variants

    @property
    def OpinionFormatter(self):
        return self.__opinion_formatter

    # endregion
