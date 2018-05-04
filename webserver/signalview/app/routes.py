from app import app
from flask import render_template, request
import psycopg2, json_tricks, os, csv
import numpy as np


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    no_records = 100
    
    user = {'username': 'Jonathan'}
    conn_string = "host='localhost' dbname='signals' user='postgres' password='secret'"
    print("Connecting to database\n -->> %s" % (conn_string))
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    cur.execute("SELECT * from signal_store.signal_live order by timestamp desc limit %i" % no_records)
    
    return render_template('signalview.html', title='SignalView', user=user, sig_data=cur)


@app.route('/graph', methods=['GET', 'POST'])
def graph():
    no_records = 100
    
    user = {'username': 'Jonathan'}
    conn_string = "host='localhost' dbname='signals' user='postgres' password='secret'"
    print("Connecting to database\n -->> %s" % (conn_string))
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    cur.execute("SELECT * from signal_store.signal_live order by timestamp desc limit %i" % no_records)
    sig_data = cur.fetchall()
    cur.execute("SELECT * from signal_store.signal_live_global")
    g_dat = cur.fetchall()
    
    return render_template('graph.html', title='SignalView', user=user, sig_data=sig_data, psd=extract_psd(g_dat, sig_data), labels=generate_labels(g_dat, sig_data))


@app.route('/inspect')
def inspect():
    id_s = request.args.get('id')
    print("Recieved request for: %s" % id_s)
    user = {'username': 'Jonathan'}
    conn_string = "host='localhost' dbname='signals' user='postgres' password='secret'"
    print("Connecting to database\n -->> %s" % (conn_string))
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    cur.execute("SELECT * from signal_store.signal_live_global")
    g_dat = cur.fetchall()
    
    cur.execute("SELECT * from signal_store.signal_live WHERE id = '%s'" % id_s)
    sig_data = cur.fetchall()
    
    sig_data_d = {}
    sig_data_d['frequency'] = sig_data[0][0]
    sig_data_d['class1'] = sig_data[0][1]
    sig_data_d['ts'] = sig_data[0][2]
    sig_data_d['hw_id'] = sig_data[0][3]
    sig_data_d['sig_id'] = sig_data[0][4]
    sig_data_d['class2'] = sig_data[0][5]
    sig_data_d['spec_data'] = json_tricks.loads(sig_data[0][6])
    sig_data_d['cf'] = sig_data[0][7]
    
    sig_data_d['spec_data'] = np.asarray(sig_data_d['spec_data'])
    
    fig = plt.figure()
    plt.pcolormesh(sig_data_d['spec_data'])
    plt.savefig('app/static/spec.png')
    plt.close(fig)
    
    return render_template('inspect.html', sig_data=sig_data)



def extract_psd(g_dat, sig_data):
    print(len(g_dat))
    dat = g_dat[0][0]
    dat = dat.replace('{','')
    dat = dat.replace('}','')
    dat_list = dat.split(',')
    
    cf = g_dat[0][1]
    fs = g_dat[0][2]
    buf_len = g_dat[0][3]
    rsr = g_dat[0][4]
    
    print(os.listdir())
    
    with open('app/static/psd.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['Frequency', 'psd'])
        for i in range(0, len(dat_list)):
            spamwriter.writerow([(fs * (i*rsr - buf_len/2)/buf_len) + cf, dat_list[i]])
    
def generate_labels(g_dat, sig_data):
    outx = "["
    outy = "["
    out_l = "["
    for i in sig_data:
        outx = outx + i[7] + ","
        outy = outy + "1,"
        out_l = out_l + '3,'
    outx = outx[:-1] + "]"
    outy = outy[:-1] + "]"
    out_l = out_l[:-1] + "]"
    return [outx, outy, out_l]
    