import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from Logistic_Model_Assess import prepare_data,feature_sel
from collections import defaultdict
import matplotlib.pyplot as plt
import time

#Getting probabilities from the logistic model
def logistic_model_pd(int_state,loan_data):
    x_train, x_test, y_train, y_test = prepare_data(int_state)
    sel_feat = feature_sel(x_train, y_train)
    log_model = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=10000,C=1)
    log_model.fit(x_train[sel_feat], np.ravel(y_train))
    log_pd=list(log_model.predict_proba(loan_data[sel_feat])[0])
    while len(log_pd)!=9:
        log_pd.insert(int_state+3,0)
    print('Logistic Model '+str(int_state)+ ' is Complete!')
    return log_pd


#Compose a 1-month transition matrix
def trans_matrix_1m(loan_data,mat_display=False):
    pd_mat=defaultdict()
    pd_mat['current -1']=[1]+[0]*8
    for i in range(0,7):
        pd_mat['current '+str(i)]=logistic_model_pd(i,loan_data)
    pd_mat['current 7'] = [0] * 8 + [1]
    pd_mat_1m=pd.DataFrame(pd_mat, index=['next '+str(i) for i in range(-1,8)])
    pd_mat_1m.round(decimals=3)
    if mat_display==True:
        print(pd_mat_1m.to_string())
    print('1-month Transition matrix is Complete!')
    return pd_mat_1m


#MCMC path of state movements
def mcmc_state_path(loan_data,int_state,month_len=6,mc_num=10000,plot=False):
    pd_mat_1m=trans_matrix_1m(loan_data)
    state_path=[]
    for j in range(0,mc_num):
        initial_state=int_state
        mc_state_path=[initial_state]
        for i in range(0,month_len):
            all_state_pd=pd_mat_1m['current '+str(initial_state)]
            next_state=np.random.choice([-1,0,1,2,3,4,5,6,7],p=all_state_pd)
            mc_state_path.append(next_state)
            initial_state=next_state
        state_path.append(mc_state_path)
    if plot==True:
        for i in state_path:
            plt.plot(i)
        plt.grid(False)
        plt.xticks(np.arange(0,month_len+1,1.0))
        plt.yticks(np.arange(-1,8,1.0))
        plt.xlabel('Month')
        plt.ylabel('Status')
        plt.show()
    print(str(month_len)+'-month MCMC State Path is Complete!')
    return state_path


#MCMC transition row after a given time horizon
def mcmc_trans_matrix(loan_data,int_state,month_len=6,mc_num=10000):
    state_path=mcmc_state_path(loan_data,int_state,month_len,mc_num)
    final_state=[i[-1] for i in state_path]
    mat=defaultdict()
    for j in range(-1,8):
        mat['Next '+str(j)]=final_state.count(j)/mc_num
    mcmc_trans_mat=pd.DataFrame(mat,index=['current '+str(int_state)])
    print(str(month_len)+'-month MCMC Transition Matrix is Complete!')
    return mcmc_trans_mat


#MC mutiplication transition matrix after a given time horizon
def multiplication_trans_matrix(loan_data,int_state=0,month_len=6,full_matrix=False):
    trans_mat=trans_matrix_1m(loan_data)
    trans_mat_np=np.array([trans_mat['current '+str(i)]for i in range(-1,8)])
    mul_trans_mat=np.linalg.matrix_power(trans_mat_np,month_len)
    mult_trans_mat=pd.DataFrame({'current '+str(i):mul_trans_mat[i+1]for i in range(-1,8)},
                                index=['next '+str(j) for j in range(-1,8)])
    print(str(month_len)+'-month Transition Matrix Multiplication is Complete!')
    if full_matrix is False:
        return mult_trans_mat['current '+str(int_state)]
    else:
        return mult_trans_mat


#Two transformation methods
def trans_matrix_transformation(loan_data,int_state,month_len=6,mc_num=10000,method='mcmc',full_matrix=False):
    if method=='mcmc':
        trans_mat=mcmc_trans_matrix(loan_data,int_state,month_len,mc_num)
    elif method=='mc multiplication':
        trans_mat=multiplication_trans_matrix(loan_data,int_state,month_len,full_matrix)
    return trans_mat.to_string()


#Transition matrix of a selected test loan data
def main():
    tic_0 = time.perf_counter()

    int_state=0
    test_num=123
    x_train, x_test, y_train, y_test = prepare_data(int_state)
    loan_data=x_test.iloc[[test_num]]
    print(trans_matrix_transformation(loan_data,int_state))

    toc_0 = time.perf_counter()
    print("Total Running Time: ", str(toc_0-tic_0), " s.")


if __name__ == '__main__':
    main()
