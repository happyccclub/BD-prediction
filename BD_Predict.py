import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
#from IPython import get_ipython
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import make_pipeline

train_path = 'train.csv'
setting_path = 'setting.csv'

local_tz ='Asia/Singapore'
train_columns = ['dt','rank','point']

def GetLocalTimeFromUTC(year, month, day, hour, minute):
    utc_time = pd.Timestamp(year=year
                            , month=month
                            , day=day
                            , hour=hour
                            , minute=minute
                            , tz = 'UTC')
    return utc_time.tz_convert(local_tz)

def GetLocalTimeNow():
    return pd.Timestamp.now(local_tz)

def RecordTrainedPath(points,duration):
    return 'train_' + str(int(points)) + '_' + str(int(duration)) +'.csv'

def NumSecond(dt):
    return dt / np.timedelta64(1, 's')

def Setting_EndTime(setting):
    return pd.to_datetime(setting.at[0,'end_time']).tz_convert(local_tz)

def Setting_StartTime(setting):
    return pd.to_datetime(setting.at[0,'start_time']).tz_convert(local_tz)

def Setting_Duration(setting):
    return NumSecond(Setting_EndTime(setting) - Setting_StartTime(setting))

def Setting_PtForLast(setting):
    return setting.at[0,'ptforlast'];

def GetNumSecondFromDTEnd(end_time):
    ret = NumSecond(end_time - GetLocalTimeNow())
    if(ret < 0):
        ret = 0;
    return ret

def ProcessData(X, setting):
    return X

#Load data from csv files and merge them
def LoadData(custom_path=''):
    if(custom_path == ''):
        train = pd.read_csv(train_path)
    else:
        train = pd.read_csv(custom_path)
    setting = pd.read_csv(setting_path)
    try:
        past_train = pd.read_csv(RecordTrainedPath(setting.at[0,'ptforlast']))
    except:
        past_train = pd.DataFrame(columns=train_columns)

    train = ProcessData(train, setting)

    past_train = ProcessData(past_train, setting)

    return train, past_train, setting

def GetXY(train):
    X = train.drop("point", axis=1)
    Y = train["point"]
    return X, Y

def ResetTraining(utc_s_year, utc_s_month, utc_s_day, utc_s_hour, utc_s_minute, utc_e_year, utc_e_month, utc_e_day, utc_e_hour, utc_e_minute, ptforlast):
    train = pd.read_csv(train_path)
    setting = pd.read_csv(setting_path)
    try:
        past_train = pd.read_csv(RecordTrainedPath(setting.at[0,'ptforlast'],Setting_Duration(setting)))
    except:
        past_train = pd.DataFrame(columns=train_columns)

    df = pd.concat([past_train, train])
    df.to_csv(RecordTrainedPath(setting.at[0,'ptforlast'],Setting_Duration(setting)), index=False)

    setting.at[0,'start_time'] = GetLocalTimeFromUTC(utc_s_year, utc_s_month, utc_s_day, utc_s_hour, utc_s_minute)
    setting.at[0,'end_time'] = GetLocalTimeFromUTC(utc_e_year, utc_e_month, utc_e_day, utc_e_hour, utc_e_minute)
    setting.at[0,'ptforlast'] = ptforlast
    setting.to_csv(setting_path, index=False)

    train = pd.DataFrame(columns=train_columns)
    train.to_csv(train_path, index=False)

def AddData(rank, point, endtime):
    train = pd.read_csv(train_path)
    train.loc[len(train)] = [GetNumSecondFromDTEnd(endtime), rank, point]
    train.to_csv(train_path, index=False)

def Train_LR(X, Y):
    lr = LinearRegression(normalize=True)
    t_dt = GetLocalTimeNow()
    lr.fit(X, Y)
    t_dt = NumSecond(GetLocalTimeNow() - t_dt)
    print("LR Training score: ", lr.score(X,Y)," [",t_dt,"s]")
    return lr

def Train_NN(X, Y):
    #regr = MLPRegressor(random_state=5, max_iter=50000, hidden_layer_sizes=(100,100,100), activation='relu', solver='adam',batch_size=1, warm_start=True)
    regr = make_pipeline(StandardScaler(with_std=False),PowerTransformer(), MLPRegressor(random_state=5, max_iter=50000, hidden_layer_sizes=(200,200,200), activation='relu', solver='adam',batch_size=1, warm_start=True))
    t_dt = GetLocalTimeNow()
    regr.fit(X, Y)
    t_dt = NumSecond(GetLocalTimeNow() - t_dt)
    print("NN Training score: ", regr.score(X,Y)," [",t_dt,"s]")
    return regr

def Train_RCV(X, Y):
    rcv = RidgeCV(alphas=[0.5,0.25])
    t_dt = GetLocalTimeNow()
    rcv.fit(X, Y)
    t_dt = NumSecond(GetLocalTimeNow() - t_dt)
    print("RCV Training score: ", rcv.score(X,Y)," [",t_dt,"s]")
    return rcv

def Train_LSVR(X, Y):
    #lsvr = LinearSVR(random_state=1 ,max_iter=1000000,loss='squared_epsilon_insensitive',dual=False)lsvr = 
    lsvr = make_pipeline(StandardScaler(with_std=False),PowerTransformer(), LinearSVR(random_state=1 ,max_iter=1000000,loss='squared_epsilon_insensitive',dual=False))
    t_dt = GetLocalTimeNow()
    lsvr.fit(X, Y)
    t_dt = NumSecond(GetLocalTimeNow() - t_dt)
    print("LSVR Training score: ", lsvr.score(X,Y)," [",t_dt,"s]")
    return lsvr

def Train_PolyLR(X, Y):
    model = make_pipeline(PolynomialFeatures(2), LinearRegression(normalize=True))
    t_dt = GetLocalTimeNow()
    model.fit(X, Y)
    t_dt = NumSecond(GetLocalTimeNow() - t_dt)
    print("PolyLR Training score: ", model.score(X,Y)," [",t_dt,"s]")
    return model

def Train_PowerLR(X, Y):
    model = make_pipeline(StandardScaler(with_std=False),PowerTransformer(), LinearRegression(normalize=True))
    t_dt = GetLocalTimeNow()
    model.fit(X, Y)
    t_dt = NumSecond(GetLocalTimeNow() - t_dt)
    print("PowerLR Training score: ", model.score(X,Y)," [",t_dt,"s]")
    return model

def Predict(model, rank, dt=0):
    return model.predict([[dt, rank]])

def MeanSquaredLogError(model, X, Y):
    X_test = X.copy()
    #prediction on sample test dataset
    predictions = model.predict(X_test)
    X_test['prediction'] = predictions
    X_test['y_test'] = Y

    #remove negative prediction because we cant score mean_squared_log_error for negative value
    X_test = X_test[(X_test['prediction'] >= 0) & (X_test['y_test'] >= 0)]

    return np.sqrt(mean_squared_log_error( X_test['y_test'], X_test['prediction'] ))

# interface
class BD_Predict:
    def __init__(self, custom_path='', drop_topten = False, reloadpast=False):
        self.Reload(custom_path,drop_topten, reloadpast)

    def Reload(self, custom_path='',drop_topten = False, reloadpast=False):
        self.train, self.past, self.setting = LoadData(custom_path)
        self.X,self.Y = GetXY(self.train)
        self.PX,self.PY = GetXY(self.past)
        if(drop_topten):
            self.Y.drop(self.X[self.X['rank'] <= 10].index, axis=0, inplace=True)
            self.X.drop(self.X[self.X['rank'] <= 10].index, axis=0, inplace=True)
            self.PY.drop(self.PX[self.PX['rank'] <= 10].index, axis=0, inplace=True)
            self.PX.drop(self.PX[self.PX['rank'] <= 10].index, axis=0, inplace=True)
        if(reloadpast):
            self.X = pd.concat([self.X,self.PX])
            self.Y = pd.concat([self.Y,self.PY])
            
        self.PrintGeneralData()
        if(self.X.shape[0] != 0):
            self.lr = Train_LR(self.X,self.Y)
            self.nn = Train_NN(self.X,self.Y)
            self.rcv = Train_RCV(self.X,self.Y)
            self.lsvr = Train_LSVR(self.X,self.Y)
            self.polylr = Train_PolyLR(self.X,self.Y)
            self.powerlr = Train_PowerLR(self.X,self.Y)

    def PrintGeneralData(self):
        print('Description')
        print(self.X.describe())
        print('Correlation')
        print(pd.concat([self.X, self.Y],axis=1).corr('kendall'))
        print()
        '''
        f = plt.figure()
        plt.matshow(self.X.corr(), fignum=f.number)
        plt.xticks(range(self.X.shape[1]), self.X.columns, rotation=45)
        plt.yticks(range(self.X.shape[1]), self.X.columns)
        cb = plt.colorbar()
        plt.title('Correlation Matrix (kendall)')
        plt.show()
        '''
        
    def AddData(self, rank, point):
        AddData(rank, point, Setting_EndTime(self.setting))

    def AddMultiData(self, start_rank, points):
        for i, point in enumerate(points):
            self.AddData(start_rank+i, point)

    def Predict(self, rank, seconds = 0):
        print("lr: ", Predict(self.lr, rank, seconds))
        print("nn: ", Predict(self.nn, rank, seconds))
        print("rcv: ", Predict(self.rcv, rank, seconds))
        print("lsvr: ", Predict(self.lsvr, rank, seconds))
        print("polylr: ", Predict(self.polylr, rank, seconds))
        print("powerlr: ", Predict(self.powerlr, rank, seconds))

    def PredictNow(self, rank):
        print("lr(now): ", Predict(self.lr, rank, GetNumSecondFromDTEnd(Setting_EndTime(self.setting))))
        print("nn(now): ", Predict(self.nn, rank, GetNumSecondFromDTEnd(Setting_EndTime(self.setting))))
        print("rcv(now): ", Predict(self.rcv, rank, GetNumSecondFromDTEnd(Setting_EndTime(self.setting))))
        print("lsvr(now): ", Predict(self.lsvr, rank, GetNumSecondFromDTEnd(Setting_EndTime(self.setting))))
        print("polylr(now): ", Predict(self.polylr, rank, GetNumSecondFromDTEnd(Setting_EndTime(self.setting))))
        print("powerlr(now): ", Predict(self.powerlr, rank, GetNumSecondFromDTEnd(Setting_EndTime(self.setting))))

    def Error(self):
        print("LR\t(MeanSquaredLogError): ", MeanSquaredLogError(self.lr, self.PX, self.PY))
        print("NN\t(MeanSquaredLogError): ", MeanSquaredLogError(self.nn, self.PX, self.PY))
        print("RCV\t(MeanSquaredLogError): ", MeanSquaredLogError(self.rcv, self.PX, self.PY))
        print("LSVR\t(MeanSquaredLogError): ", MeanSquaredLogError(self.lsvr, self.PX, self.PY))
        print("PolyLR\t(MeanSquaredLogError): ", MeanSquaredLogError(self.polylr, self.PX, self.PY))
        print("PowerLR\t(MeanSquaredLogError): ", MeanSquaredLogError(self.powerlr, self.PX, self.PY))

    def NewEvent(self, utc_s_year, utc_s_month, utc_s_day, utc_s_hour, utc_s_minute, utc_e_year, utc_e_month, utc_e_day, utc_e_hour, utc_e_minute, ptforlast):
        ResetTraining(utc_s_year, utc_s_month, utc_s_day, utc_s_hour, utc_s_minute, utc_e_year, utc_e_month, utc_e_day, utc_e_hour, utc_e_minute, ptforlast)
        self.Reload()

    def TimeLeft(self):
        return Setting_EndTime(self.setting) - GetLocalTimeNow()

    def PlotRank(self, seconds=-1):
        #get_ipython().run_line_magic('matplotlib', 'inline')
        if seconds == -1:
            seconds = GetNumSecondFromDTEnd(Setting_EndTime(self.setting))
        show = 20000
        interval = 100
        n = int(show/interval) + 1
        data = np.zeros((7,n))
        for i in range (n):
            data[0,i] = show-i*interval;
            data[1,i] = Predict(self.lr, data[0,i], seconds)
            data[2,i] = Predict(self.nn, data[0,i], seconds)
            data[3,i] = Predict(self.rcv, data[0,i], seconds)
            data[4,i] = Predict(self.lsvr, data[0,i], seconds)
            data[5,i] = Predict(self.polylr, data[0,i], seconds)
            data[6,i] = Predict(self.powerlr, data[0,i], seconds)

        # plot the index for the x-values
        plt.figure()
        plt.plot(data[0], data[1], label='LR')
        plt.plot(data[0], data[2], label='NN')
        plt.plot(data[0], data[3], label='RCV')
        plt.plot(data[0], data[4], label='LSVR')
        plt.plot(data[0], data[5], label='PolyLR')
        plt.plot(data[0], data[6], label='PowerLR')
        plt.xlabel('Rank')
        plt.ylabel('Points')
        plt.title('PlotRank')
        plt.legend()
        plt.show()
        return data

    def PlotTime(self, rank=5000, from_now=False):
        #get_ipython().run_line_magic('matplotlib', 'inline')
        if from_now:
            max_time = GetNumSecondFromDTEnd(Setting_EndTime(self.setting))
        else:
            max_time = Setting_Duration(self.setting)
        interval = 900
        n = int(max_time/interval) + 1
        data = np.zeros((7,n))
        for i in range (n):
            data[0,i] = max_time-i*interval;
            data[1,i] = Predict(self.lr, rank, data[0,i])
            data[2,i] = Predict(self.nn, rank, data[0,i])
            data[3,i] = Predict(self.rcv, rank, data[0,i])
            data[4,i] = Predict(self.lsvr, rank, data[0,i])
            data[5,i] = Predict(self.polylr, rank, data[0,i])
            data[6,i] = Predict(self.powerlr, rank, data[0,i])

        # plot the index for the x-values
        plt.figure()
        plt.plot(data[0], data[1], label='LR')
        plt.plot(data[0], data[2], label='NN')
        plt.plot(data[0], data[3], label='RCV')
        plt.plot(data[0], data[4], label='LSVR')
        plt.plot(data[0], data[5], label='PolyLR')
        plt.plot(data[0], data[6], label='PowerLR')
        plt.xlabel('Timeleft')
        plt.ylabel('Points')
        plt.title('PlotTime')
        plt.legend()
        plt.show()
        return data

    def Plot3D_Data(self):
        #get_ipython().run_line_magic('matplotlib', 'qt5')
        ax = plt.axes(projection='3d')
        X = self.X['dt']
        Y = self.X['rank']
        Z = self.Y
        ax.scatter3D(X,Y,Z, 'gray')
        ax.set_xlabel('dt')
        ax.set_ylabel('rank')
        ax.set_zlabel('point')
        plt.show()

    def Plot3D_Z(self, model='nn',from_now=False):
        #get_ipython().run_line_magic('matplotlib', 'qt5')

        n = 500
        rank_x = np.zeros((n,))
        for i in range (n):
            rank_x[i] = (40*i);

        if from_now:
            max_time = GetNumSecondFromDTEnd(Setting_EndTime(self.setting))
        else:
            max_time = Setting_Duration(self.setting)
        n = int(max_time/600) + 1
        time_x = np.zeros((n,))
        for i in range (int(max_time/600) + 1):
            time_x[i] = i*600;

        X, Y = np.meshgrid(time_x,rank_x)
        Z = np.zeros(X.shape)
        if(model=='nn'):
            input_model = self.nn
        elif  (model=='lr'):
            input_model = self.lr
        elif  (model=='rcv'):
            input_model = self.rcv
        elif  (model=='lsvr'):
            input_model = self.lsvr
        elif  (model=='polylr'):
            input_model = self.polylr
        elif  (model=='powerlr'):
            input_model = self.powerlr
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                    Z[i,j] = Predict(input_model, Y[i,j], X[i,j])
                    #print(Z[i,j],X[i,j],Y[i,j])
            print('\rPercentage: ',i/X.shape[0],end='',flush=True)
        print('[DONE]')
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        X = self.X['dt']
        Y = self.X['rank']
        Z = self.Y
        ax.scatter3D(X,Y,Z, 'gray')
        
        ax.set_xlabel('dt')
        ax.set_ylabel('rank')
        ax.set_zlabel('point')
        plt.show()

# test
target_rank = 10000
bd = BD_Predict(drop_topten=True)
rank_points = bd.PlotRank(0)
time_points = bd.PlotTime(target_rank,from_now=False)
#
print('Target Rank: ',target_rank)
print('Time Left: ',bd.TimeLeft(),'[',NumSecond(bd.TimeLeft()),'s]')
bd.PredictNow(target_rank)
bd.Predict(target_rank)

'''
model = make_pipeline(PolynomialFeatures(2), LinearRegression(normalize=True))
model.fit(bd.X, bd.Y)
print('test_model: ',model.score(bd.X, bd.Y))
print('test_model predict: ',Predict(model, 5000))

time = GetLocalTimeFromUTC(2020, 6, 29, 6 ,59)
local_time = GetLocalTimeNow()

delta = time - local_time

'''