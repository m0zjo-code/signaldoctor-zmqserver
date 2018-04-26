import zmq, argparse
import psycopg2
import sys, json
from signaldoctorlib import id_generator


def main(args):
    if args.d:
        # Socket to talk to IQ server
        port_rx = 5557
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

        cur.execute("TRUNCATE signal_store.signal_live")
        conn.commit()

        while True:
            input_packet = socket_rx.recv_pyobj()
            try:
                cf = str(input_packet['metadata']['cf'])
                radiotype = str(input_packet['metadata']['radio'])
                cur.execute("INSERT INTO signal_store.signal_live (id, freq, class1, class2, timestamp, radioid, freq_offset) VALUES (%s, %s, %s, %s, %s, %s, %s)", (id_generator(size=24), cf, input_packet['pred1'], input_packet['pred2'], input_packet['timestamp'], radiotype, input_packet['offset']))
                conn.commit()
            except psycopg2.ProgrammingError:
                cur.execute("INSERT INTO signal_store.signal_live (id, class1, class2, timestamp, freq_offset) VALUES (%s, %s, %s, %s, %s, %s)", (id_generator(size=6), input_packet['pred1'], input_packet['pred2'], input_packet['timestamp'], input_packet['offset']))
                conn.commit()
        cur.close()
        conn.close()
    print("No arguments supplied - exiting. For help please run with -h flag")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='### Signal Logging Server for Signal Doctor [MAIN SIGNALS]- Jonathan Rawlinson 2018 ###', epilog="For more help please see the docs")
    
    parser.add_argument('-d', help='RX from classifier with default settings (in->127.0.0.1:5557) - target database signal_store.signal_live on localhost with login "postgres" and password "secret"', action="store_true")
    
    args = parser.parse_args()
    main(args)
