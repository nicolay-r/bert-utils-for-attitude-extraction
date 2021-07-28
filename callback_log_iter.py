from arekit.contrib.experiment_rusentrel.evaluation.results.two_class import TwoClassEvalResult

PARAMS_SEP = u"; "
NAME_VALUE_SEP = u': '


def create_iteration_verbose_eval_msg(eval_result, data_type, epoch_index):
    assert (isinstance(eval_result, TwoClassEvalResult))
    title = u"Stat for [{dtype}], e={epoch}:".format(dtype=data_type, epoch=epoch_index)
    contents = [u"{doc_id}: {result}".format(doc_id=doc_id, result=result)
                for doc_id, result in eval_result.iter_document_results()]
    return u'\n'.join([title] + contents)


def create_iteration_short_eval_msg(eval_result, data_type, epoch_index):
    assert (isinstance(eval_result, TwoClassEvalResult))
    title = u"Stat for '[{dtype}]', e={epoch}".format(dtype=data_type, epoch=epoch_index)
    params = [u"{m_name}{nv_sep}{value}".format(m_name=metric_name, nv_sep=NAME_VALUE_SEP, value=round(value, 4))
              for metric_name, value in eval_result.iter_total_by_param_results()]
    contents = PARAMS_SEP.join(params)
    return u'\n'.join([title, contents])


def parse_epochs_count(filepath):
    epochs = None
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if u'Stat for' in line:
                epochs = line.split(u'=')[-1]
    return int(epochs)


def parse_last(filepath, col):
    """ Stat for '[DataType.Train]', e=11
        f1: 0.96; f1_pos: 0.97; f1_neg: 0.94; pos_prec: 0.97; neg_prec: 0.95; pos_recall: 0.97; neg_recall: 0.94
    """
    assert(isinstance(col, unicode))

    last = 0
    with open(filepath, 'r') as f:
        for line in f.readlines():
            for p in line.split(PARAMS_SEP):
                if col not in p:
                    continue
                value = p.split(NAME_VALUE_SEP)[1]
                last = float(value)
    return last