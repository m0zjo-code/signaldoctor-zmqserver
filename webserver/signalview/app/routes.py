from app import app
from flask import render_template, request
import psycopg2



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
