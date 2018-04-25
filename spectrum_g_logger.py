import zmq
import psycopg2
import sys, json
from signaldoctorlib import id_generator



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
    cur.execute("INSERT INTO signal_store.signal_live_global (recent_psd) VALUES (%s)", (input_packet['recent_psd'],)) ### Remember comma!
    conn.commit()

cur.close()
conn.close()