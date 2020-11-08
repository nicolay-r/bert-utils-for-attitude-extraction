from arekit.contrib.bert.core.input.io_utils import BertIOUtils


class CustomBertIOUtils(BertIOUtils):

    def get_experiment_sources_dir(self):
        return u"output"
