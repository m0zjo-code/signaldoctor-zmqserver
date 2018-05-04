import zmq
import psycopg2
import sys, json, argparse
from signaldoctorlib import id_generator



def main(args):
    if args.d:
        # Socket to talk to IQ server
        port_rx = 5558
        context_rx = zmq.Context()
        socket_rx = context_rx.socket(zmq.SUB)
        print("Collecting updates from IQ server...")
        socket_rx.connect ("tcp://127.0.0.1:%i" % port_rx)
        socket_rx.setsockopt_string(zmq.SUBSCRIBE, "")
        conn_string = "host='localhost' dbname='signals' user='postgres' password='secret'"
        print("Connecting to database\n -->> %s" % (conn_string))

        conn = psycopg2.connect(conn_string)

        cur = conn.cursor()
        print("Connected!\n")

        while True:
            input_packet = socket_rx.recv_pyobj()
            cur.execute("TRUNCATE signal_store.signal_live_global")
            cur.execute("INSERT INTO signal_store.signal_live_global (recent_psd, cf, fs, buf_len, resampling_ratio) VALUES (%s, %s, %s, %s, %s)", (input_packet['recent_psd'],input_packet['cf'],input_packet['fs'],input_packet['buf_len'],input_packet['resampling_ratio'],)) ### Remember comma!
            conn.commit()

        cur.close()
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='### Signal Logging Server for Signal Doctor [SCENARIO META-DATA] - Jonathan Rawlinson 2018 ###', epilog="For more help please see the docs")
    
    parser.add_argument('-d', help='RX from processor with default settings (in->127.0.0.1:5558) - target database signal_store.signal_live_global on localhost with login "postgres" and password "secret"', action="store_true")
    
    args = parser.parse_args()
    main(args)