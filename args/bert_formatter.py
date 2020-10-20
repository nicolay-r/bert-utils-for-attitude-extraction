from arekit.contrib.bert.formatters.sample.formats import SampleFormatters
from args.base import BaseArg


class BertFormatterArg(BaseArg):

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--bert-input-fmt',
                            dest='bert_input_fmt',
                            type=unicode,
                            nargs=1,
                            choices=list(SampleFormatters.iter_supported()),
                            default=1,
                            help='Formatter according to the paper ...')

    @staticmethod
    def read_argument(args):
        return args.bert_input_fmt[0]
