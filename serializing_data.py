import logging
from os import path
from os.path import dirname, join

from arekit.common.experiment.cv.sentence_based import SentenceBasedCVFolding
from arekit.common.experiment.data.serializing import SerializationData
from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.experiment.cv.doc_stat.rusentrel import RuSentRelDocStatGenerator

from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.io_utils import RuSentiFramesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollection

from arekit.processing.lemmatization.mystem import MystemWrapper


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BertRuSentRelBasedSerializaingData(SerializationData):

    def __init__(self, terms_per_context, labels_scaler=None):
        assert(isinstance(labels_scaler, BaseLabelScaler) or labels_scaler is None)

        # TODO. Provide rusentrel version.

        logger.info("Create experiment [{}]".format(str(labels_scaler)))

        super(BertRuSentRelBasedSerializaingData, self).__init__(
            labels_scale=ThreeLabelScaler() if labels_scaler is None else labels_scaler)

        self.__stemmer = MystemWrapper()
        # TODO. Provide this from the outside
        self.__synonym_collection = RuSentRelSynonymsCollection.load_collection(
            stemmer=self.__stemmer,
            is_read_only=True)
        self.__opinion_formatter = RuSentRelOpinionCollectionFormatter(self.__synonym_collection)
        self.__cv_folding_algorithm = SentenceBasedCVFolding(
            docs_stat=RuSentRelDocStatGenerator(synonyms=self.__synonym_collection,
                                                # TODO. Provide rusentrel version
                                                version=RuSentRelVersions.V11),
            docs_stat_filepath=path.join(self.get_data_root(), u"docs_stat.txt"))

        # TODO. Provide this from the outside
        self.__frames_collection = RuSentiFramesCollection.read_collection(version=RuSentiFramesVersions.V10)
        self.__unique_frame_variants = FrameVariantsCollection.create_unique_variants_from_iterable(
            variants_with_id=self.__frames_collection.iter_frame_id_and_variants(),
            stemmer=self.__stemmer)

        self.__terms_per_context = terms_per_context

        self.__sources_dir = None
        self.__results_dir = None

    # region public properties

    @property
    def Stemmer(self):
        return self.__stemmer

    @property
    def SynonymsCollection(self):
        return self.__synonym_collection

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
    def CVFoldingAlgorithm(self):
        return self.__cv_folding_algorithm

    @property
    def TermsPerContext(self):
        return self.__terms_per_context

    # endregion

    # region public methods

    def set_experiment_results_dir(self, value):
        assert(isinstance(value, unicode))
        self.__results_dir = value

    def set_experiment_sources_dir(self, value):
        assert(isinstance(value, unicode))
        self.__sources_dir = value

    def get_data_root(self):
        return path.join(dirname(__file__), u"data/")

    def get_experiment_sources_dir(self):
        src_dir = self.__sources_dir
        if self.__sources_dir is None:
            # Considering a source dir by default.
            src_dir = join(self.get_data_root())
        return src_dir

    def get_experiment_results_dir(self):
        if self.__results_dir is None:
            # Considering the same as a source dir
            return self.get_experiment_sources_dir()
        return self.__results_dir

    # endregion
