#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, arghelper
from arghelper import inputfile, inputdir, check_range
import os, sys
from functions import *


class CycaspParser(argparse.ArgumentParser):    
    ''' Parser class '''
    def error(self, message):
        sys.stderr.write('> error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def main(argv=None):
    ''' Handles the parsing of arguments '''
    parser = CycaspParser(
        prog='CYCASP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
----------------------------------------------
CYCASP - Copyright (c) Georges Hattab
Under the MIT License (MIT)
----------------------------------------------
            ''',
        epilog='''
Usage examples:
./cycasp.py -i img_directory/ -g 100
./cycasp.py -f filename.csv -t 2
            ''')

    # Arguments to be handled
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.5\n')

    parser.add_argument('-i', '--input', default=None, metavar='dir',\
                        help='run CYCASP on the supplied directory',\
                        type=lambda x: arghelper.inputdir(parser, x))

    parser.add_argument('-f', '--file', default=None,\
                        help='run CYCASP on CSV file', type=inputfile)
                        
    parser.add_argument('-d', \
                        help='particle diameter (odd int >3 px, default 11)',\
                        nargs="?", default=11, type=check_range)
                        
    parser.add_argument('-e', \
                        help='euclidean distance (default 10)',\
                        nargs="?", default=10, type=int)
    
    parser.add_argument('-r', \
                        help='red channel differences (default 50)',\
                        nargs="?", default=50, type=int)
                        
    parser.add_argument('-g', \
                        help='green channel differences (default 50)',\
                        nargs="?", default=50, type=int)

    parser.add_argument('-b', \
                        help='blue channel differences (default 50)',\
                        nargs="?", default=50, type=int)
                        
    parser.add_argument('-t', \
                        help='merge time window (default 10)',\
                        nargs="?", default=10, type=int)

    try:
        args = parser.parse_args()
        if (args.file == None) and (args.input == None):
           parser.error('--input or --file must be supplied')

        elif (args.file != None) and (args.input != None):
            parser.error('please choose either an input dir. or a csv file')

        elif args.file != None:
            print parser.description, "\n"
            print 'Input file', args.file
            print 'Particle diamater: d=%s(px)' %args.d
            print 'User thresholds: e=%s(px), r=%s, g=%s, b=%s, t=%s frames.' %(args.e, args.r, args.g, args.b, args.t)
            st = time.time()
            d = load_data(args.file)
            elapsed_time(st)
            d, G = modalgo(d, args.d, args.e, args.r, args.g, args.b, args.t)
            elapsed_time(st)
            export_data(d, G)
            elapsed_time(st)
            print "Press Enter to exit"
            raw_input()

        elif args.input != None:
            if not args.input.endswith("/"):
                parser.error('please supply a suitable directory (Usage examples below).')
            else:
                print parser.description, "\n"
                print 'Input directory', args.input
                print 'Particle diamater: d=%s(px)' %args.d
                print 'User thresholds: e=%s(px), r=%s, g=%s, b=%s, t=%s frames.' %(args.e, args.r, args.g, args.b, args.t)
                st = time.time()
                outdir, red, green, blue = preprocess(args.input)
                elapsed_time(st)
                d = get_data(outdir, red, green, blue, args.d)
                elapsed_time(st)
                d, G = modalgo(d, args.d, args.e, args.r, args.g, args.b, args.t)
                elapsed_time(st)
                export_data(d, G)
                elapsed_time(st)
                print "Press Enter to exit"
                raw_input()

    except IOError, msg:
        parser.error(str(msg))

    except KeyboardInterrupt:
        parser.exit(1, "\nExecution aborted")


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main(sys.argv[1:])

