from arekit.common.entities.formatters.types import EntityFormattersService, EntityFormatterTypes
from arekit.common.experiment.data_type import DataType
from arekit.contrib.bert.samplers.types import SampleFormattersService, BertSampleFormatterTypes


class Common:

    log_dir = u"log/"
    __log_eval_iter_filename_template = u"cb_eval_{iter}_{dtype}.log"

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
