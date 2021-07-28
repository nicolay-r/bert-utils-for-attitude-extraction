from arekit.common.entities.formatters.types import EntityFormattersService, EntityFormatterTypes
from arekit.common.experiment.data_type import DataType
from arekit.contrib.bert.samplers.types import SampleFormattersService, BertSampleFormatterTypes
from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.experiment_rusentrel.labels.scalers.two import TwoLabelScaler
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions


class Common:

    log_dir = u"log/"
    __log_eval_iter_filename_template = u"cb_eval_{iter}_{dtype}.log"

    # tags that utilized in finetuned model names.
    __tags = {
        None: None,
        RuAttitudesVersions.V12: u'ra-12',
        RuAttitudesVersions.V20Base: u'ra-20b',
        RuAttitudesVersions.V20BaseNeut: u'ra-20bn',
        RuAttitudesVersions.V20Large: u'ra-20l',
        RuAttitudesVersions.V20LargeNeut: u'ra-20ln'
    }

    @staticmethod
    def create_labels_scaler(labels_count):
        assert (isinstance(labels_count, int))

        if labels_count == 2:
            return TwoLabelScaler()
        if labels_count == 3:
            return ThreeLabelScaler()

        raise NotImplementedError("Not supported")

    @staticmethod
    def __combine_tag_with_full_model_name(full_model_name, tag):
        assert(isinstance(tag, unicode))
        return u'-'.join([full_model_name, tag])

    # region public methods

    @staticmethod
    def get_tag_by_ruattitudes_version(version):
        if version not in Common.__tags:
            return None
        return Common.__tags[version]

    @staticmethod
    def iter_tag_values():
        return Common.__tags.itervalues()

    @staticmethod
    def iter_tag_keys():
        return Common.__tags.iterkeys()

    @staticmethod
    def combine_tag_with_full_model_name(full_model_name, tag):
        assert(isinstance(tag, unicode) or tag is None)
        return Common.__combine_tag_with_full_model_name(
                full_model_name=full_model_name,
                tag=unicode(tag))

    @staticmethod
    def create_log_eval_filename(iter_index, data_type):
        assert(isinstance(iter_index, int))
        assert(isinstance(data_type, DataType))
        return Common.__log_eval_iter_filename_template.format(iter=iter_index, dtype=data_type)

    @staticmethod
    def create_full_model_name(sample_fmt_type, entities_fmt_type, labels_count):
        assert(isinstance(sample_fmt_type, BertSampleFormatterTypes))
        assert(isinstance(entities_fmt_type, EntityFormatterTypes))
        assert(isinstance(labels_count, int))

        full_model_name = u"bert-{sample_fmt}-{entities_fmt}-{labels_mode}l".format(
            sample_fmt=SampleFormattersService.type_to_name(sample_fmt_type),
            entities_fmt=EntityFormattersService.find_name_by_type(entities_fmt_type),
            labels_mode=int(labels_count))

        return full_model_name

    # endregion
