from arekit.contrib.networks.core.io_utils import NetworkIOUtils


class BertIOUtils(NetworkIOUtils):

    # We conside to save serialized results into model dir,
    # rather than experiment dir in a base implementation,
    # as model affects on text_b, entities representation, etc.
    @staticmethod
    def get_target_dir(experiment):
        """ Provides a main directory for input
        """
        return experiment.DataIO.get_model_root(experiment.Name)
