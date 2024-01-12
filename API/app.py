import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from datetime import datetime
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
from numpy import log
import statsmodels.api as sm
from pmdarima.arima.utils import ndiffs
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.emplike import elregress

from fastapi import FastAPI
from mangum import Mangum
import uvicorn

if not firebase_admin._apps:
  cred = credentials.Certificate("bill-extractor-586a1-firebase-adminsdk-bnhw2-c376f19014.json")
  firebase_admin.initialize_app(cred)
  
db = firestore.client()

userid = "" #"tl5YWL9orwgWfMftlUgzDSDpm9Q2"
datacollectioname = "" #'expenses'
futuredatacolname = "" #'futurevalues'
para = "" #"all"
len_predic = 0 #15
needminlen = 8 # Data need minimum 
doclimit = 200 #200
key = ""

catlist = ["Transport","Housing_and_Bills","Medical","Personal_Care","Food","Grocery"]

# Create variables for catagory
for c in catlist:
  exec(str(c) + "_valsdic" + " = {}")
  exec(str(c) + "_vals" + " = []")

def dataarrange(colname,useridv,paraname="all"):
  for cl in catlist:
    eval(str(cl) + "_vals").clear()
      
  users_ref = db.collection(colname)
  docs = users_ref.stream()

  dcount = 0

  if(paraname == "all"):
    for doc in docs:
      if(doc.to_dict()['id'] == useridv):
        dcount = dcount + 1
        if(dcount <= doclimit):
          for k in doc.to_dict().keys():
            if(k in catlist):
              try:
                eval(str(k) + "_valsdic")[doc.to_dict()['Date']] = float(doc.to_dict()[k])
              except:
                None
  else:
    if(paraname not in catlist):
      return "Error: Catagory Not Matched" ## Errors
    else:
      for doc in docs:
        if(doc.to_dict()['id'] == useridv):
          dcount = dcount + 1
          if(dcount <= doclimit):
            try:
              eval(str(paraname) + "_valsdic")[doc.to_dict()['Date']] = float(doc.to_dict()[paraname])
            except:
              None

  for c in catlist:
    ordered_data = sorted( eval(str(c) + "_valsdic").items(), key = lambda x:datetime.strptime(x[0], '%d/%m/%Y'), reverse=False)
    val = [ordered_data[i][1] for i in range(0,len(ordered_data))]
    for i in val:
      eval(str(c) + "_vals").append(float(i))

def getnoofdays(catv,len_predic):
  lend = int(len_predic*2) # check last len_predic*n dates
  dataval = eval(str(catv) + "_vals")
  dataval = dataval[-lend:]
  tdatalen = len(dataval)
  nonzdatalen = np.count_nonzero(dataval)
  if(tdatalen <= len_predic):
    nodays = nonzdatalen
  else:
    nodays = int(round((nonzdatalen / tdatalen) * len_predic,0)) 
  return nodays,tdatalen,nonzdatalen

def addtodatabase(futuredatacolname,userid,data):
  users_ref = db.collection(futuredatacolname)
  users_ref.document(userid).delete()
  users_ref.document(userid).set(data)


def predict_newdata(parameter,len_predic):
  dval = eval(str(parameter) + "_vals")
  if(all(v == 0 for v in dval)):
    return "Error: All data are zeros" ## Errors
  else:
    data = pd.DataFrame(dval, columns=['Amount'])
    data = data.dropna()

    #########

    # dropping the null values
    adfresult = adfuller(data.Amount.dropna())

    ########

    ### D Value find ####
    if(adfresult[1] > 0.05):
      d = ndiffs(data.Amount.dropna(), test='adf')
    else:    
      d = 0


    #### P and Q
    if(d==0):
      dataforpq = data.Amount.dropna()
    elif(d==1):
      dataforpq = data.Amount.diff().dropna()
    elif(d==2):
      dataforpq = data.Amount.diff().diff().dropna()
    elif(d==3):
      dataforpq = data.Amount.diff().diff().diff().dropna()
    else:
      dataforpq = data.Amount.diff().diff().diff().diff().dropna()

    try:
      pacf, ci = sm.tsa.pacf(dataforpq, alpha=0.05)
    except:
      pacf=[0]

    try:
      acf, ci = sm.tsa.acf(dataforpq, alpha=0.05)
    except:
      acf=[0]

    p = max(acf)
    q = max(pacf)

    if(pd.isna(p)):
      p = 0

    if(pd.isna(q)):
      q = 0

    #########

    # ARIMA Model define
    ARIMA_model = sm.tsa.arima.ARIMA(data.Amount, order=(p,d,q))

    # Training process 
    model_fit = ARIMA_model.fit()

    ############
    
    fc = model_fit.forecast(steps=len_predic, alpha=0.05)
    return fc

app = FastAPI()
handler = Mangum(app)

@app.get("/")
async def index():
    return "Home"

@app.get("/api")
async def student_data(user_id:str,colexpname:str,colstorename:str,parameters:str,prediclen:int,token:str):

    userid = user_id #"tl5YWL9orwgWfMftlUgzDSDpm9Q2"
    datacollectioname = colexpname #'expenses'
    futuredatacolname = colstorename #'futurevalues'
    para = parameters #"all"
    len_predic = prediclen #15
    needminlen = 8 # Data need minimum 
    key = "wRqJLyjCz5O9FjU"
        
    if(key == token):
        f1 = dataarrange(datacollectioname,userid,para)

        if f1 is not None:
            data = {"error":f1}
            addtodatabase(futuredatacolname,userid,data)
        else:
            data = {}
            if(para == 'all'):
                totalval = {}
                for cat in catlist:
                    noofd,tdatalen,nonzdatalen = getnoofdays(cat,len_predic)
                    datav = {}
                    if len(eval(str(cat) + "_vals")) <= needminlen:
                        datav = {"error":"Error: Data Not Enough"} ## Send error message
                        totalval[cat] = {"Total":datav,"Considered Dates":noofd,"Total Available Dates":tdatalen, "Non-Zero Dates":nonzdatalen, "Predict Dates":len_predic}
                    else:
                        valslist = predict_newdata(cat,len_predic)
                        if all(isinstance(s, str) for s in valslist):
                            datav = {"error":valslist} ## Send error message
                        else:
                            cat_budget_tlt = round(valslist[0:noofd].sum(),2)
                            totalval[cat] = {"Total":cat_budget_tlt,"Considered Dates":noofd,"Total Available Dates":tdatalen, "Non-Zero Dates":nonzdatalen, "Predict Dates":len_predic}
                            for i,v in enumerate(valslist):
                                if(v>0):
                                    datav[str(i)] = round(v, 2)
                                else:
                                    datav[str(i)] = 0           
                    data[cat] = datav
                data["Total_Budgets"] = totalval
                addtodatabase(futuredatacolname,userid,data)
            else:
                totalval = {}
                datav = {}
                noofd,tdatalen,nonzdatalen = getnoofdays(cat)
                if len(eval(str(para) + "_vals")) <= needminlen:
                    datav = {"error":"Error: Data Not Enough"} ## Send error message
                else:
                    valslist = predict_newdata(para,len_predic)
                    if all(isinstance(s, str) for s in valslist):
                        datav = {"error":valslist} ## Send error message
                        totalval[para] = {"Total":datav,"Considered Dates":noofd,"Total Available Dates":tdatalen, "Non-Zero Dates":nonzdatalen, "Predict Dates":len_predic}
                    else:
                        cat_budget_tlt = round(valslist[0:noofd].sum(),2)
                        totalval[para] = {"Total":cat_budget_tlt,"Considered Dates":noofd,"Total Available Dates":tdatalen, "Non-Zero Dates":nonzdatalen, "Predict Dates":len_predic}
                        for i,v in enumerate(valslist):
                            if(v>0):
                                datav[str(i)] = round(v, 2)
                            else:
                                datav[str(i)] = 0 
                data = {para:datav}
                data["Total_Budgets"] = totalval
                addtodatabase(futuredatacolname,userid,data)
        return {"Status":"OK"}
    else:
       return {"Status":"Error"}

    
if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)
