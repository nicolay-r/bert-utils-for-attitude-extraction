from arekit.common.experiment.data.serializing import SerializationData
from arekit.common.frame_variants.collection import FrameVariantsCollection

from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
from bert_model_io import BertModelIO


class CustomSerializationData(SerializationData):

    def __init__(self, labels_scaler, stemmer, frames_version, model_io):
        assert(isinstance(model_io, BertModelIO))

        super(CustomSerializationData, self).__init__(labels_scaler, stemmer)

        self.__model_io = model_io
        self.__opinion_formatter = RuSentRelOpinionCollectionFormatter()
        self.__frames_collection = RuSentiFramesCollection.read_collection(version=frames_version)
        self.__unique_frame_variants = FrameVariantsCollection.create_unique_variants_from_iterable(
            variants_with_id=self.__frames_collection.iter_frame_id_and_variants(),
            stemmer=self.__stemmer)

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
    def Stemmer(self):
        return self.__stemmer

    # endregion
