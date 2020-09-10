import logging
from os import path
from os.path import dirname, join

from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.common.experiment.cv.default import SimpleCVFolding
from arekit.common.experiment.cv.sentence_based import SentenceBasedCVFolding
from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.experiment.cv.doc_stat.rusentrel import RuSentRelDocStatGenerator
from arekit.common.experiment.data_io import DataIO

from arekit.processing.lemmatization.mystem import MystemWrapper

from arekit.source.embeddings.rusvectores import RusvectoresEmbedding
from arekit.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.source.rusentiframes.io_utils import RuSentiFramesVersions
from arekit.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
from arekit.source.rusentrel.synonyms import RuSentRelSynonymsCollection

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RuSentRelBasedExperimentsIOUtils(DataIO):

    def __init__(self, labels_scaler=None, init_word_embedding=True):
        assert(isinstance(labels_scaler, BaseLabelScaler) or labels_scaler is None)

        logger.info("Create experiment [{}]".format(str(labels_scaler)))

        super(RuSentRelBasedExperimentsIOUtils, self).__init__(
            labels_scale=ThreeLabelScaler() if labels_scaler is None else labels_scaler)

        self.__stemmer = MystemWrapper()
        self.__synonym_collection = RuSentRelSynonymsCollection.load_collection(
            stemmer=self.__stemmer,
            is_read_only=True)
        self.__opinion_formatter = RuSentRelOpinionCollectionFormatter
        self.__word_embedding = self.__create_word_embedding() if init_word_embedding else None
        self.__cv_folding_algorithm = self.__init_sentence_based_cv_folding_algorithm()

        self.__frames_collection = RuSentiFramesCollection.read_collection(version=RuSentiFramesVersions.V10)
        self.__unique_frame_variants = FrameVariantsCollection.create_unique_variants_from_iterable(
            variants_with_id=self.__frames_collection.iter_frame_id_and_variants(),
            stemmer=self.__stemmer)

        self.__evaluator = TwoClassEvaluator(self.__synonym_collection)
        self.__callback = None

        self.__model_io = None

        self.__sources_dir = None
        self.__results_dir = None

    # region public properties

    @property
    def Stemmer(self):
        return self.__stemmer

    @property
    def ModelIO(self):
        return self.__model_io

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
    def WordEmbedding(self):
        return self.__word_embedding

    @property
    def OpinionFormatter(self):
        return self.__opinion_formatter

    @property
    def Evaluator(self):
        return self.__evaluator

    @property
    def CVFoldingAlgorithm(self):
        return self.__cv_folding_algorithm

    @property
    def Callback(self):
        return self.__callback

    # endregion

    # region private methods

    def __create_word_embedding(self):
        we_filepath = path.join(self.get_data_root(), u"w2v/news_rusvectores2.bin.gz")
        logger.info("Loading word embedding: {}".format(we_filepath))
        return RusvectoresEmbedding.from_word2vec_format(filepath=we_filepath,
                                                         binary=True)

    def __init_sentence_based_cv_folding_algorithm(self):
        return SentenceBasedCVFolding(
            docs_stat=RuSentRelDocStatGenerator(synonyms=self.__synonym_collection),
            docs_stat_filepath=path.join(self.get_data_root(), u"docs_stat.txt"))

    def __init_simple_cv_folding_algoritm(self):
        return SimpleCVFolding()

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
            experiments_name = u'rusentrel'
            src_dir = join(self.get_data_root(), u"./{}/".format(experiments_name))
        return src_dir

    def get_experiment_results_dir(self):
        if self.__results_dir is None:
            # Considering the same as a source dir
            return self.get_experiment_sources_dir()
        return self.__results_dir

    def get_word_embedding_filepath(self):
        return path.join(self.get_data_root(), u"w2v/news_rusvectores2.bin.gz")

    # endregion
