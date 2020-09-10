import os
from arekit.common.utils import create_dir_if_not_exists
from arekit.networks.nn_io import NeuralNetworkModelIO


class CustomNeuralNetworkIO(NeuralNetworkModelIO):

    def __init__(self):
        self.__model_root = None
        self.__model_name = None

    def set_model_root(self, value):
        assert(isinstance(value, unicode))
        self.__model_root = value

    def set_model_name(self, value):
        assert(isinstance(value, unicode))
        self.__model_name = value

    @property
    def ModelSavePathPrefix(self):
        return os.path.join(self.__get_model_states_dir(),
                            u'{}'.format(self.__model_name))

    def __get_model_states_dir(self):
        result_dir = os.path.join(self.__model_root, os.path.join(u'model_states/'))
        create_dir_if_not_exists(result_dir)
        return result_dir


