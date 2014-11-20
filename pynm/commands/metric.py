# -*- coding:utf-8 -*-

import csv

import numpy

from pynm.metric.itml import learn_metric, convert_data


class ItmlCommand:
    name = 'itml'
    help = 'Information Theoretic Metric Learning'

    @classmethod
    def build_arg_parser(cls, parser):
            parser.add_argument('-i',
                                '--input_data',
                                default=None,
                                type=str,
                                metavar='FILE',
                                required=True,
                                help='input data file')
            parser.add_argument('-l',
                                '--input_labels',
                                default=None,
                                type=str,
                                metavar='FILE',
                                help='input labels file')
            parser.add_argument('-p',
                                '--input_pairs',
                                default=None,
                                type=str,
                                metavar='FILE',
                                help='input pairs file')
            parser.add_argument('-o',
                                '--output_data',
                                default=None,
                                type=str,
                                metavar='FILE',
                                help='output data file')
            parser.add_argument('-m',
                                '--output_metric',
                                default=None,
                                type=str,
                                metavar='FILE',
                                help='output metric file')
            parser.add_argument('-w',
                                '--output_weights',
                                default=None,
                                type=str,
                                metavar='FILE',
                                help='output weights file')

            parser.add_argument('-d',
                                '--delimiter',
                                default='\t',
                                type=str,
                                metavar='DELIM',
                                help='delimiter (default: "\\t")')
            parser.add_argument('-s',
                                '--sparse',
                                action='store_true',
                                help='sparse format (not implemented yet)')
            parser.add_argument('--header',
                                action='store_true',
                                help='has header')

            parser.add_argument('-U',
                                '--u_param',
                                default=1.0,
                                type=float,
                                metavar='DISTANCE',
                                help='U parameter (max distance for same labels, default: 1.0)')
            parser.add_argument('-L',
                                '--l_param',
                                default=1.0,
                                type=float,
                                metavar='DISTANCE',
                                help='L parameter (min distance for different labels, default: 1.0)')
            parser.add_argument('-S',
                                '--slack',
                                default=1.0,
                                type=float,
                                metavar='SLACK',
                                help='slack variable (default: 1.0)')
            parser.add_argument('-N',
                                '--max_iteration_number',
                                default=1000,
                                type=int,
                                metavar='MAX',
                                help='max iteration (default: 1000)')

    def run(self, args):
        with open(args.input_data) as in_:
            header, data = self.load_data(in_,
                                          delimiter=args.delimiter,
                                          has_header=args.header)
        if args.input_labels is not None:
            with open(args.input_labels) as in_:
                labels = self.load_labels(in_)
                pairs = None
        elif args.input_pairs is not None:
            with open(args.input_pairs) as in_:
                pairs = self.load_pairs(in_)
                labels = None

        metric = learn_metric(data,
                              labels=labels,
                              pairs=pairs,
                              u=args.u_param,
                              l=args.l_param,
                              slack=args.slack,
                              max_iter=args.max_iteration_number,
                              is_sparse=args.sparse)

        if args.output_metric is not None:
            with open(args.output_metric, 'w') as o_:
                self.export_metric(o_, metric, header)
        if args.output_weights is not None:
            weights = numpy.diag(metric)
            with open(args.output_weights, 'w') as o_:
                self.export_weights(o_, weights, header)
        if args.output_data is not None:
            converted_data = convert_data(metric, data)
            with open(args.output_data, 'w') as o_:
                self.export_data(o_, converted_data, header)
        return 0

    def load_data(self,
                  input_data,
                  delimiter='\t',
                  has_header=False):
        reader = csv.reader(input_data, delimiter=delimiter)
        if has_header:
            header = {value: key for key, value in enumerate(reader.next())}
        else:
            header = None

        data = []
        for row in reader:
            data.append(numpy.array(list(map(lambda x: float(x), row))))
        return header, data

    def load_labels(self, input_labels):
        return list(map(lambda x: int(x), input_labels))

    def load_pairs(self, input_pairs, delimiter='\t', header=None):
        pairs = []
        if header is None:
            for line in input_pairs:
                row = line.split(delimiter)
                idx1 = int(row[0])
                idx2 = int(row[1])
                similar = int(row[2]) > 0
                pairs.append((idx1, idx2, similar))
        else:
            for line in input_pairs:
                row = line.split(delimiter)
                idx1 = header[row[0]]
                idx2 = header[row[1]]
                similar = int(row[2]) > 0
                pairs.append((idx1, idx2, similar))
        return pairs

    def export_metric(self,
                      output,
                      metric,
                      header=None,
                      sparse=False):
        if sparse:
            raise NotImplementedError('sparse is not supported yet.')

        writer = csv.writer(output)
        if header is not None:
            writer.writerow(header)
        for row in metric:
            writer.writerow(row)

    def export_weights(self,
                       output,
                       weights,
                       header=None):
        writer = csv.writer(output)
        if header is not None:
            writer.writerow(header)
        writer.writerow(weights)

    def export_data(self,
                    output,
                    data,
                    header=None,
                    sparse=False):
        if sparse:
            raise NotImplementedError('sparse is not supported yet.')

        writer = csv.writer(output)
        if header is not None:
            writer.writerow(header)
        for row in data:
            writer.writerow(row)


class MetricCommand:
    name = 'metric'
    help = 'Metric Learning'

    sub_commands = [ItmlCommand]

    def build_arg_parser(self, parser):
        subparsers = parser.add_subparsers(title='algorithm', dest='algorithm')
        for command in self.sub_commands:
            subparser = subparsers.add_parser(command.name, help=command.help)
            command.build_arg_parser(subparser)
        self._parser = parser

    def run(self, args):
        if args.algorithm is None:
            self._parser.print_help()
            return 1
        sub_command = self._get_sub_command(args.algorithm)
        return sub_command.run(args)

    def _get_sub_command(self, algorithm):
        return next(filter(lambda x: x.name == algorithm, self.sub_commands))()
