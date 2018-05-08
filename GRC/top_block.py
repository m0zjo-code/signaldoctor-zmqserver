#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Top Block
# Generated: Tue May  8 14:13:29 2018
##################################################

from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import zeromq
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import osmosdr
import time


class top_block(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Top Block")

        ##################################################
        # Variables
        ##################################################
        self.vector_size = vector_size = 2**20
        self.samp_rate = samp_rate = 1000000

        ##################################################
        # Blocks
        ##################################################
        self.zeromq_pub_sink_0 = zeromq.pub_sink(gr.sizeof_gr_complex, vector_size, 'tcp://127.0.0.1:5555', 100, False, -1)
        self.osmosdr_source_0 = osmosdr.source( args="numchan=" + str(1) + " " + 'hackrf=0' )
        self.osmosdr_source_0.set_sample_rate(samp_rate)
        self.osmosdr_source_0.set_center_freq(100e6, 0)
        self.osmosdr_source_0.set_freq_corr(0, 0)
        self.osmosdr_source_0.set_dc_offset_mode(2, 0)
        self.osmosdr_source_0.set_iq_balance_mode(2, 0)
        self.osmosdr_source_0.set_gain_mode(True, 0)
        self.osmosdr_source_0.set_gain(14, 0)
        self.osmosdr_source_0.set_if_gain(32, 0)
        self.osmosdr_source_0.set_bb_gain(30, 0)
        self.osmosdr_source_0.set_antenna('', 0)
        self.osmosdr_source_0.set_bandwidth(0, 0)
          
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, vector_size)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_stream_to_vector_0, 0), (self.zeromq_pub_sink_0, 0))    
        self.connect((self.osmosdr_source_0, 0), (self.blocks_stream_to_vector_0, 0))    

    def get_vector_size(self):
        return self.vector_size

    def set_vector_size(self, vector_size):
        self.vector_size = vector_size

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.osmosdr_source_0.set_sample_rate(self.samp_rate)


def main(top_block_cls=top_block, options=None):

    tb = top_block_cls()
    tb.start()
    try:
        raw_input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
