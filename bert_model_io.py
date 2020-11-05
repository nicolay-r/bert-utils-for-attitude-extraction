from os.path import join

from arekit.common.model.model_io import BaseModelIO


class BertModelIO(BaseModelIO):

    target_dir = u"output"

    def __init__(self, full_model_name):
        assert(isinstance(full_model_name, unicode))
        self.__full_model_name = full_model_name

    def get_model_dir(self):
        return join(BertModelIO.target_dir, self.__full_model_name)
