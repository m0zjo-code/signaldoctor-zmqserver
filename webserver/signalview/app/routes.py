from app import app
from flask import render_template, request
import psycopg2, json, os, csv



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
    psd = cur.fetchall()[0]
    
    return render_template('graph.html', title='SignalView', user=user, sig_data=sig_data, psd=extract_psd(psd, sig_data))

def extract_psd(psd, sig_data):
    dat = psd[0]
    dat = dat.replace('{','')
    dat = dat.replace('}','')
    dat_list = dat.split(',')
    
    print(os.listdir())
    
    with open('app/static/psd.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        for i in range(0, len(dat_list)):
            spamwriter.writerow([i, dat_list[i]])
    