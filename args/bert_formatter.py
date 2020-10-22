from arekit.contrib.bert.supported import SampleFormattersService
from args.base import BaseArg


class BertFormatterArg(BaseArg):

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--bert-input-fmt',
                            dest='bert_input_fmt',
                            type=unicode,
                            nargs=1,
                            choices=list(SampleFormattersService.iter_supported(True)),
                            default=1,
                            help='Formatter according to the paper ...')

    @staticmethod
    def read_argument(args):
        type_value = args.bert_input_fmt[0]
        return SampleFormattersService.find_type_by_value(type_value)
