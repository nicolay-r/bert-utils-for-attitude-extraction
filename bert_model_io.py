from arekit.common.model.model_io import BaseModelIO


class BertModelIO(BaseModelIO):

    def __init__(self, full_model_name):
        assert(isinstance(full_model_name, unicode))
        self.__full_model_name = full_model_name

    def get_model_name(self):
        return self.__full_model_name
