import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import matplotlib
import math
import matplotlib as mpl
import matplotlib.image as mpimg
import networkx as nx
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy import ndimage
import matplotlib.lines as mlines
import random

##plotting pre-setup
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=22)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams["figure.figsize"] = 8,8

#Normalize data, filter out the low values
#Turn values lower than threshold into zeros, renormalize the sum to 1
#Input: dataframe, the threshold value
#Output: normalized dataframe
def normalize(gp,threshold):
    gp[gp<threshold]=0
#     display(gp)
    data_nor = gp.mul(1/gp.sum(), axis=1)
    return data_nor

#Count the appearance of glycofeatures in the glycans.
#Input: glycan, motif in linearcode format. (df.index format works) A flag to choose binary/count match
#Output: A dataframe with glycans as rows and glycofeatures as columns.
def get_motif_counts(glycan_linearcode,motif_linearcode,binary_count=True):  #ADD .copy()

    # Preparation: get the motifs' linearcode in the right format for following contains() and count()
    # Input: the list of motif_linearcode
    # Output: fixed format of the motif linearcodes
    def fix_motif_format(motifs):
        for i in range(len(motifs)):
            m = motifs[i]
            if m in ['(*2', '(*3', '(*4','(*3&GNb4','(*3&GNb6','Mb4GNb4(Fa6)GN']:
                continue
            else:
                indexes = []
                for j,char in enumerate(m):
                    if char in ['(',')']:
                        indexes.append(j)
                indexes.reverse()
                for ind in indexes:
                    m = m[:ind] + '\\' + m[ind:]
            motifs[i] = m
        return motifs

    # Preparation: Add brackets to the glycans for branch counting
    # Input: list of glycans (in character format)
    # Output: list of bracketed glycans
    def fix_glycan_format(glycans):
        glycan_llc = []
        for g in glycans:
            if ';' in g:
                ind = g.find(';')
                g='('+g[:ind]+')'+g[ind:]
            else:
                g='('+g+')'
            glycan_llc.append(g)
        return glycan_llc

    motif_linearcode_fix = fix_motif_format(motif_linearcode.copy())
    glycan_linearcode_fix = fix_glycan_format(glycan_linearcode.copy())
    lm = pd.DataFrame(index = glycan_linearcode_fix, columns = motif_linearcode_fix)

    # Get the branches
    nofuc = lm.index.str.replace('\(Fa6\)','', regex=True) # this will not change lm.index
    nofuc = nofuc.str.replace('\(Fa3\)','', regex=True)
    nofuc = nofuc.str.replace('\(Fa2\)','', regex=True)
    nofucbisect = nofuc.str.replace('\(GNb4\)','', regex=True)
    nofucbisectsia=nofucbisect.str.replace('\(NNa3\)','', regex=True)
    nofucbisectsia=nofucbisectsia.str.replace('\(NNa6\)','', regex=True)
    branching = nofucbisectsia.str.count('\(')
    # print(branching)
    is_triantennary  = (branching==3).astype(int)
    is_biantennary = (branching==2).astype(int)
    is_tetraantennary = (branching==4).astype(int)
#     print(lm.index)
#     print(nofucbisectsia)

    for glycofeature in lm.columns:
        #branching motifs
        if glycofeature == '(*2': 
            lm [glycofeature] = is_biantennary
        elif glycofeature == '(*3&GNb4': 
            lm [glycofeature] = is_triantennary&lm.index.str.contains('GNb4\)|GNb4\(')
        elif glycofeature == '(*4': 
            lm [glycofeature] = is_tetraantennary
        elif glycofeature == '(*3&GNb6': 
            lm [glycofeature] = is_triantennary

        # #repeating monosacchride motifs
        elif '*' in glycofeature:
            repeat_count = int(glycofeature.split('*')[1])
            repeat_motif = glycofeature.split('*')[0]
            motifcount_arr = []
            for glycan in lm.index.array:
                countt = len([*re.finditer(repeat_motif+r'[a-b]',glycan)])#if search '(', need to be '\('
                motifcount_arr.append(countt)
            motifcount_arr = pd.Series(motifcount_arr)
            lm[glycofeature]  = (motifcount_arr==repeat_count).astype(int).array

        # #internal motifs
        elif not glycofeature.startswith('\('):
#             print(glycofeature)
            if glycofeature[0] != "(":
                final_count = []
                for glycan in glycan_linearcode_fix:#lm.index.to_numpy():
                    count=0
                    while len(glycan)>0:
                        find_i = glycan.find(glycofeature)
#                         print('in ',glycan,' found feature ',glycofeature, ' at index ', find_i)
                        if find_i == -1:
                            glycan = ''
                            break
                        elif glycofeature[0]=='F' or glycan[find_i-1] != '(':
                            count+=1
                        glycan = glycan[find_i+len(glycofeature)-1:]
                    final_count.append(count)
#                 print(final_count)
                lm[glycofeature] = final_count

        # #Get the terminal ones
        else:
            lm[glycofeature] = lm.index.str.count(glycofeature)

    # exclude triantennary count that are GNb4, prevent double counting
    lm.loc[lm['(*3&GNb4'] ==1,'(*3&GNb6'] = 0
    # Fix the format of index and columns
    lm.index = glycan_linearcode
    lm.columns = motif_linearcode
    if binary_count:
        lm[lm>1]=1 
    return lm


#Get lp simulation. DOUBLE CHECK: columns/rows names have to be identical!!
#Input: three dataframes, glycan to motif counts, lectin to motif binding, and glycoprofile (glycan to sample)
#Output: simulated lectin profile
def simulate_lp(glycan_motif,lec_motif,gp,norm=True):
    glycan_lec = glycan_motif.dot(lec_motif.T)
    lp = (glycan_lec.T).dot(gp)
    if norm:
        lp = lp/lp.sum()
    return lp

#Plot multiple lectin profiles as a barplot. Separated by sample. Can handle two samples.
#Input: Dataframe of the experimental/control lp, a list of lp dfs in same lectin rows to compare, legend name for the list dfs.
#Output: ax of two plots. 
def plot_lp(control_lp, valid_dfs=[], valid_colnames=[],sample_1='Fet',sample_2='IgG'):
    FETUINB_df = pd.DataFrame(0.0, index = control_lp.index, columns =['Experiment']+valid_colnames)#,'Kiki_iso'])
    FETUINB_df['Experiment']=control_lp.iloc[:,:int(len(control_lp.columns)/2)].mean(1)
    IGG_df = pd.DataFrame(0.0, index = control_lp.index, columns =['Experiment']+ valid_colnames)#,'Kiki_Manno'])
    IGG_df['Experiment']=control_lp.iloc[:,int(len(control_lp.columns)/2):].mean(1)
    for i in range(len(valid_dfs)):
        Fet_columns = valid_dfs[i].columns[valid_dfs[i].columns.str.contains(sample_1)]
        IgG_columns = valid_dfs[i].columns[valid_dfs[i].columns.str.contains(sample_2)]
        FETUINB_df[valid_colnames[i]]=valid_dfs[i][Fet_columns].mean(1)
        IGG_df[valid_colnames[i]]=valid_dfs[i][IgG_columns].mean(1)
    # display(FETUINB_df)
    axFet = FETUINB_df.plot.bar(title=sample_1,rot=45,figsize=(12, 8))
    axFet.legend(bbox_to_anchor=(1, 0.75))
    axIgG = IGG_df.plot.bar(title=sample_2,rot=45,figsize=(12, 8))
    axIgG.legend(bbox_to_anchor=(1, 0.75))
    plt.show()
    return axFet,axIgG

#Generate the polt that can be used in papers, manuscripts
#Input: simply experimental lp and simulated lp
#Output: a plot, should be able to save, not coded yet
def plot_lp_formal(exp_lp,sim_lp,ylimit=0.5,rmse_flag=True,filename='LP'):
    def compute_rmse(exp_gp, pred_df):
        rmses = []
        for _, column_data in pred_df.items():
            mse = mean_squared_error(exp_gp, column_data)
            rmse = math.sqrt(mse)
            rmses.append(rmse)
        return sum(rmses) / len(rmses)

    # Get the bar data for angelo's exp data
    Fet_exp_df = exp_lp.iloc[:, :int(len(exp_lp.columns) / 2)]
    Fet_exp_mean = Fet_exp_df.mean(1).to_numpy()
    Fet_exp_std = Fet_exp_df.std(1).to_numpy()
    IgG_exp_df = exp_lp.iloc[:, int(len(exp_lp.columns) / 2):]
    IgG_exp_mean = IgG_exp_df.mean(1).to_numpy()
    IgG_exp_std = IgG_exp_df.std(1).to_numpy()

    # Get the simulation data
    Fet_sim_df = sim_lp.iloc[:, :int(len(sim_lp.columns) / 2)]
    Fet_sim_mean = Fet_sim_df.mean(1).to_numpy()
    IgG_sim_df = sim_lp.iloc[:, int(len(sim_lp.columns) / 2):]
    IgG_sim_mean = IgG_sim_df.mean(1).to_numpy()

    # Create lists for the plot
    materials = exp_lp.index.to_numpy()
    x_pos = np.arange(len(materials))

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

    # First subplot
    ax1.bar(x_pos, Fet_exp_mean, yerr=Fet_exp_std, align='center', alpha=1, color='gray', ecolor='grey', capsize=5, width=0.25, label='Experiment')
    ax1.bar(x_pos + 0.25, Fet_sim_mean, align='center', alpha=1, color='w', edgecolor='grey', capsize=5, width=0.25, label='Simulation')
    ax1.set_ylabel('Fraction of total\nabundance')
    ax1.set_xticks(x_pos + 0.125)
    ax1.set_ylim(0, ylimit)
    ax1.set_title('rhA1AT')
    ax1.set_xticklabels(materials, rotation=45)

    # Add data points as dots
    for i, material in enumerate(materials):
        # Experimental data points
        ax1.scatter([i] * len(Fet_exp_df.columns), Fet_exp_df.loc[material], color='black', alpha=0.5, s=10, zorder=10)
        # Simulation data points
        ax1.scatter([i + 0.25] * len(Fet_sim_df.columns), Fet_sim_df.loc[material], color='black', alpha=0.5, s=10, zorder=10)

    # Second subplot
    ax2.bar(x_pos, IgG_exp_mean, yerr=IgG_exp_std, align='center', alpha=1, color='gray', ecolor='grey', capsize=5, width=0.25)
    ax2.bar(x_pos + 0.25, IgG_sim_mean, align='center', alpha=1, color='w', edgecolor='grey', capsize=5, width=0.25)
    ax2.set_xticks(x_pos + 0.125)
    ax2.set_ylim(0, ylimit)
    ax2.set_title('pdA1AT')
    ax2.set_xticklabels(materials, rotation=45)

    # Add data points as dots
    for i, material in enumerate(materials):
        # Experimental data points
        ax2.scatter([i] * len(IgG_exp_df.columns), IgG_exp_df.loc[material], color='black', alpha=0.5, s=10, zorder=10)
        # Simulation data points
        ax2.scatter([i + 0.25] * len(IgG_sim_df.columns), IgG_sim_df.loc[material], color='black', alpha=0.5, s=10, zorder=10)

    if rmse_flag:
        FetB_rmse = compute_rmse(Fet_sim_df.iloc[:, 0], Fet_exp_df)
        ax1.text(0.95, 0.95, 'RMSE={:.2e}'.format(FetB_rmse), verticalalignment='top', horizontalalignment='right', transform=ax1.transAxes)
        IgG_rmse = compute_rmse(IgG_sim_df.iloc[:, 0], IgG_exp_df)
        ax2.text(0.95, 0.95, 'RMSE={:.2e}'.format(IgG_rmse), verticalalignment='top', horizontalalignment='right', transform=ax2.transAxes)

    # Shared y-tick labels and legend
    plt.yticks(np.arange(0, ylimit + 0.1, 0.1))
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('../Figure/'+filename+'.png')
    plt.show()

#Linear regression function 
#Input: raw data!! simLP and expLP.
#Output: regression df.
def get_reg_rule(sim_lp_raw,exp_lp_raw,sample_1='Fet',sample_2='IgG'):
    # y = pd.concat([exp_lp_raw.iloc[:,:int(len(exp_lp_raw.columns)/2)].mean(1), exp_lp_raw.iloc[:,int(len(exp_lp_raw.columns)/2):].mean(1)], axis=1)
    sample1_cols = exp_lp_raw.columns[exp_lp_raw.columns.str.contains(sample_1)]
    sample2_cols = exp_lp_raw.columns[exp_lp_raw.columns.str.contains(sample_2)]
    # print(sample1_cols)
    # print(exp_lp_raw[sample1_cols])
    y = pd.concat([exp_lp_raw[sample1_cols].mean(1),exp_lp_raw[sample2_cols].mean(1)], axis=1)
    # X = sim_lp_raw[['FetB1','IgG1']] # Put the best simulation here.
    X = pd.concat([sim_lp_raw[sample1_cols].mean(1),sim_lp_raw[sample2_cols].mean(1)], axis=1)
    y.columns = [sample_1,sample_2]
    X.columns = [sample_1,sample_2]
    reg_fit_df = pd.DataFrame(0.0,columns=['coef','intercept'],index=exp_lp_raw.index)
    # reg_fit_df.drop(labels=['GNL'],inplace=True)##remember to take off if keep GNL
    for ind in X.index:
        X_ = X.loc[ind].to_numpy().reshape(-1,1)
        y_ = y.loc[ind].to_numpy()
        # display(X_)
        linr_model = LinearRegression().fit(X_, y_)
        reg_fit_df.loc[ind] = [linr_model.coef_[0],linr_model.intercept_]
    # # Some modification to the reg_fit_df. Regardless for now
    fix_index=reg_fit_df['coef']<=0
    reg_fit_df.loc[fix_index,'coef']=0.1
    # reg_fit_df.loc[fix_index,'intercept']=y.loc[fix_index].mean(1)
    # reg_fit_df.loc['RCA-I']=[0.298,0]
    # reg_fit_df.to_excel('RegFit_rule_12lec_grant.xlsx',index = True)
    return reg_fit_df

#Correct simulation with linear regression factors.
#Input: raw sim LP, a df of regression coefficient and intercept. (lectins row should be the same)
#Output: fixed lp.
def fit_sim_lp_reg_rules(sim_lp_raw, reg_rule):
    if (sim_lp_raw.index != reg_rule.index).all:
        reg_rule=reg_rule.loc[sim_lp_raw.index]
    fit_lp = sim_lp_raw.mul(reg_rule['coef'].to_numpy(),axis=0)
    fit_lp = fit_lp.add(reg_rule['intercept'].to_numpy(),axis=0)
    fit_lp[fit_lp<0]=0
    return fit_lp

# Use old model code
def build_model(size, depth, input_size, output_size,ran_seed=None):
    # build model
    if ran_seed:
        tf.random.set_seed(ran_seed)
        np.random.seed(ran_seed)
        random.seed(ran_seed)
    model_shape = [keras.Input(shape = (input_size,))]
    for i in np.arange(depth):
        model_shape.append(tf.keras.layers.Dense(size,activation="relu", name="layer"+str(i)))# kernel_initializer='zero',
        if i==(depth-1): #last layer
            model_shape.append(tf.keras.layers.Dense(output_size, activation="softmax", name="prediction"))
    model = keras.Sequential(model_shape)
    model = compile_model(model)
    print(model.summary())
    return model

def compile_model(model):
    model.compile(
        optimizer= keras.optimizers.RMSprop(),# 'Adam',
        loss=keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    return model

#Function to train and save the model
#Outputs a .npz file that rmse, prediction, sim prediction and glycan order
#If record_all == True, the npz file will record all replicates' prediction
def train_model_best(train_lp,test_lp,train_gp,test_gp,log_name='New_test',nrepeat=10,test_SIMlp=None,record_all=False,ran_seed=None):
    Nglycan = len(train_gp.index)
    Nlectin = len(test_lp.index) 
    loss_arr = []
    rmse_arr = []
    best_rmse = 100
    best_pred = []
    best_simpred=[]
    seeds_list=[]
    seeds = [None]*10
    if ran_seed:
        random.seed(ran_seed)
        seeds = [random.randint(0, 1000) for _ in range(10)]

    save_path='../Log/'+log_name+'/'
    isExist = os.path.exists(save_path)
    if isExist:
        print(f"The LOG '{log_name}' already exists. Stopping the execution to avoid overwrite.")
        return
    else:
        os.makedirs(save_path)
    for i in range(nrepeat):
        #shuffle samples in the train set
        train_lp_ran=train_lp.sample(frac=1,random_state=i,axis=1)
        train_gp_ran=train_gp[train_lp_ran.columns]

        train_x = tf.constant(train_lp_ran.T.astype('float64'))
        test_x = tf.constant(test_lp.transpose().astype('float64'))
        train_y = tf.constant(train_gp_ran.transpose().astype('float64'))
        test_y = tf.constant(test_gp.transpose().astype('float64'))
        if test_SIMlp is not None:
            test_xSIM = tf.constant(test_SIMlp.transpose().astype('float64'))
        
        model_seed = seeds[i]
        seeds_list.append(model_seed)
        model = build_model(20,4,Nlectin, Nglycan,model_seed)
        history = model.fit(train_x,train_y,batch_size=4,epochs=500) # history will be used to plot the loss vs epoch
        # history_list.append(history.history)
        (loss,rmse) = model.evaluate(test_x, test_y, batch_size=4)
        loss_arr.append(loss)
        rmse_arr.append(rmse)
        
        if record_all == False:
            if rmse < best_rmse:
                best_rmse = rmse
                model.save_weights(save_path+'best')
                best_pred=model.predict(test_x)
                if test_SIMlp is not None:
                    best_simpred=model.predict(test_xSIM)
                # best_repeat = math.ceil(nfold/5)-1
        else:
            model.save_weights(save_path+'repeat'+str(i+1))
            pred=model.predict(test_x)
            best_pred.append(pred)
            if test_SIMlp is not None:
                simpred=model.predict(test_xSIM)
                best_simpred.append(simpred)
                
    np.savez(save_path+'Results',
             rmse_arr=np.array(rmse_arr,dtype=object),
             best_pred=np.array(best_pred,dtype=object),
             best_SIMpred =np.array(best_simpred,dtype=object),
             glycan_array = test_gp.index,
             seeds_list = seeds_list
            )
    return seeds_list

# load data function
#Note: the output gp does not have the same glycan order 
def load_result_npz(log,gp,norm=True,record_all=False):
    result_npz = np.load('../Log/'+log+'/Results.npz',allow_pickle=True)
    rmse = result_npz['rmse_arr']
    pred = result_npz['best_pred']
    SIMpred = result_npz['best_SIMpred']
    all_glycans_real = result_npz['glycan_array']
    if record_all == False:
        best_pred_df = pd.DataFrame(pred.T,index=all_glycans_real,columns=gp.columns)
        best_SIMpred_df = pd.DataFrame(SIMpred.T,index=all_glycans_real,columns=gp.columns)
        best_SIMpred_df.sort_index(axis=1,inplace=True)
    else:
        best_pred_df = pd.DataFrame(index=all_glycans_real)
        best_SIMpred_df = pd.DataFrame(index=all_glycans_real)
        # print(pred.shape)
        for i in range(pred.shape[0]):
            temp_pred_df = pd.DataFrame(pred[i].T,index=all_glycans_real,columns=gp.columns)
            temp_SIMpred_df = pd.DataFrame(SIMpred[i].T,index=all_glycans_real,columns=gp.columns)
            best_pred_df = pd.concat([best_pred_df, temp_pred_df], axis=1)
            best_SIMpred_df = pd.concat([best_SIMpred_df, temp_SIMpred_df], axis=1)
        pass
    if norm:
        best_pred_df = normalize(best_pred_df,0.01)
        best_SIMpred_df = normalize(best_SIMpred_df,0.01)
    gp = gp.loc[all_glycans_real]
    return rmse, best_pred_df, gp,best_SIMpred_df

# Melt the observed and predict data into one df for later use (plot, compare)
# Takes either single pred_df or a list of multiple pred_df
# Input: prediction dataframe or a list of df, gp, names for the list of df (count must align # of df)
# Output: the melt df, with column names: glycan, CL(sample), observed, predict, diff
def melt_for_plot(pred_df,gp,name=[]):
    gp['glycan']=gp.index
    compare_df = gp.melt(id_vars=['glycan'], value_vars=gp.columns[:-1], var_name='sample', value_name='Observed')
    if isinstance(pred_df,list):
        temperate_df = None
        for i,pred in enumerate(pred_df):
            pred.index=gp.index
            pred = pred.loc[:,gp.columns[:-1]]
            pred['glycan']=gp.index
            replicate_columns = pred.columns[:-1] #important step to keep the melt df align
            if temperate_df is None:
                temperate_df = pred.melt(id_vars=['glycan'], value_vars=replicate_columns, var_name='sample', value_name='Predicted')
            else:
                temperate_df[i]=pred.melt(id_vars=['glycan'], value_vars=replicate_columns)['value']
        #fix the column names
        temperate_df = temperate_df.set_axis(['glycan', 'sample']+name, axis=1)

        compare_df = compare_df.merge(temperate_df, on=['glycan', 'sample'], how='left')
        
    else:
        pred_df.index=gp.index
        pred_df = pred_df.loc[:,gp.columns[:-1]]
        pred_df['glycan'] = gp.index
        #Get a melted prediction df
        replicate_columns = pred_df.columns[:-1]
        pred_df_melted = pred_df.melt(id_vars=['glycan'], value_vars=replicate_columns, var_name='sample', value_name='Predicted')
        compare_df = compare_df.merge(pred_df_melted, on=['glycan', 'sample'], how='left')

    return compare_df


# Plot the melt df into bar plot, threshold can be set to filter low values when plotting.
# Input: the melt df from function(melt_for_plot), and optional a threshold for filtering tiny abundances. 
# Output: the melt melt df used for ploting with column names: glycan, data(source), abundance 
# Print: the barplot, linearcode for the glycans shown on the plot. 
# Note any value above the threshold is kept. 
def plot_glycoprofile(df, threshold=0.05, rot=90, legend_loc='best', color=sns.color_palette(), ylimit=None, filename='predGP'):
    test = df.copy()
    test.index = test['glycan']
    test = test.melt(id_vars=['glycan'], value_vars=test.columns[2:], var_name='data', value_name='abundance')
    
    # Only drop the glycans that have a mean lower than the threshold
    for glycan in test.glycan.unique():
        glycan_rows = test.loc[test.glycan == glycan]
        for datafrom in glycan_rows.data.unique():
            glycan_data_rows = glycan_rows.loc[glycan_rows.data == datafrom]
            if glycan_data_rows.abundance.mean() < threshold:
                test.drop(glycan_data_rows.index, inplace=True)
    print(test['glycan'].unique())
    
    def standard_error(x):
        return np.std(x) / np.sqrt(len(x))
    #Print the std error values
    summary = test.groupby(['glycan', 'data']).agg({'abundance': ['mean', standard_error]}).reset_index()
    summary.columns = ['glycan', 'data', 'mean_abundance', 'standard_error']
    std_error_mean = summary.groupby(['data']).agg({'standard_error': ['mean']}).reset_index()
    print(std_error_mean)

    fig, ax = plt.subplots(figsize=(len(test['glycan'].unique()) * 0.7, 5))
    g = sns.barplot(data=test, x='glycan', y='abundance', hue='data', err_kws={'linewidth': 0.3}, palette=color)
    
    sns.despine(top=True, right=True)
    
    if ylimit is not None:
        plt.ylim(0, ylimit)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=legend_loc)
    plt.xticks(rotation=rot)
    # Adjust layout to fit all elements
#     plt.tight_layout()
    # Save the figure with tight bounding box
    plt.savefig('../Figure/' + filename + '.png', bbox_inches='tight')

    return test


# Plot the melt df into bar plot, but arranged follow mz values in order
# Input: the melt df from function(melt_for_plot), a dataframe containing glycan and mz values, and optional a threshold for filtering tiny abundances. 
# Output: 
# Print: mz values that have been plotted, sorted in ascending order
def plot_glycoprofile_formal(melt_df,mz_df,threshold=0.01,title='',bar_width=15,bar_color=None,xlimDIY=None):
#Convert the glycans from linear code to their mz values
    mz_df = mz_df['mz'].to_dict()
    melt_df[melt_df.columns[2:]] = melt_df[melt_df.columns[2:]].astype('float')
    melt_df['mz'] = melt_df['glycan'].map(mz_df)
    melt_df.index=melt_df['mz']
    melt_df.drop(labels=['mz'],axis=1,inplace=True)
    plot_sum_iso_df = pd.DataFrame()
    #Add up values under same mz for every sample(gp)
    for sample in melt_df['sample'].unique():
        temp_df = melt_df.loc[melt_df['sample']==sample].copy()  # make a copy
        #for isomers having same mz, sum them up 
        temp_df = temp_df.groupby(temp_df.index).sum(numeric_only=True)
        #apply plot threshold to the data
        numeric_cols = temp_df.select_dtypes(include=[np.number])
        numeric_cols[numeric_cols<threshold]=0
        numeric_cols = numeric_cols.mul(1/numeric_cols.sum(), axis=1)
        temp_df.loc[:, numeric_cols.columns] = numeric_cols  # use .loc to avoid SettingWithCopyWarning
        #concat into one dataframe
        plot_sum_iso_df = pd.concat([plot_sum_iso_df,temp_df])
        
    plot_sum_iso_df.index = pd.Series(plot_sum_iso_df.index).round(decimals=0).astype(int)
    #Get standard error (ste) for error bar
    def standard_error(x):
        return np.std(x) / np.sqrt(len(x))
    #Calculate mean and ste for plot
    grouped = plot_sum_iso_df.groupby(plot_sum_iso_df.index).agg(['mean',standard_error])

    plt.figure(figsize=(15,8))
    width = bar_width
    plotted_mz = []
    for i, column in enumerate(plot_sum_iso_df.columns):
        #get non-zero rows within grouped[column, 'mean']
        non_zero_mz = grouped.index[grouped[column, 'mean']>0]
        plotted_mz = list(set(plotted_mz).union(set(non_zero_mz)))#update if every iteration
        # Plot means with error bars, shift the bars for each column by i*width
        if bar_color == None:
            plt.bar(grouped.index + (i-1)*width, grouped[column, 'mean'], yerr=grouped[column, 'standard_error'], 
                capsize=2, width=width, label=column)
        else:
            plt.bar(grouped.index + (i-1)*width, grouped[column, 'mean'], yerr=grouped[column, 'standard_error'], 
                capsize=2, width=width, label=column,color=bar_color[i])
        # print('ste for '+column+' is: ')
        # print(grouped[column, 'standard_error'].to_numpy().astype(str))
        print('average ste for '+column+' is: '+str(grouped[column, 'standard_error'].mean()))
    plt.xlabel('m/z')
    plt.ylabel('Fraction of total abundance')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')#,loc='best')
    
    plotted_mz.sort()
    print(plotted_mz)
    lower_bound = (plotted_mz[0] // 500) * 500  # This rounds down
    upper_bound = math.ceil(plotted_mz[-1] / 500.0) * 500 
    if xlimDIY is not None:
        plt.xlim(xlimDIY)
    else:
        plt.xlim(lower_bound,upper_bound)
    plt.show()
    return plot_sum_iso_df,plotted_mz


#Calculate the prediction accuracy based on the rate of True Positive rate
#Input: glycoprofile and prediction
#Output: a float number representing the accuracy
def get_accuracy_binary(gp,pred):
    gp[gp>0] = 1
    pred[pred>0] = 1
    count = len(gp.index)
    TP_TN = (gp == pred).sum()
    return TP_TN/count


#Calculate rmse between one column and all columns in a df.
#Input: a series and a dataframe
#Outout: average rmse calculated
def compute_rmse(exp_gp,pred_df):
    rmses = []
    for _, column_data in pred_df.items():
        mse = mean_squared_error(exp_gp, column_data)
        rmse = math.sqrt(mse)
        rmses.append(rmse)
    return sum(rmses) / len(rmses)

#Calculate accuracy (binary) between one column and all columns in a df. On True Positive rate
#average the pred_gp, cut-off threshold, then get the accuracy
#Input: a series and a dataframe
#Outout: average rmse calculated
def compute_accuracy(exp_gp,pred_gp,thres):
    pred_mean = pred_gp.mean(1)
    pred_mean[pred_mean < thres] = 0
    if pred_mean.sum() != 0:
        pred_final = pred_mean / pred_mean.sum()
    else:
        print('Threshold too big. Nothing left.')
        return
    pred_detected = pred_final > 0
    actual_detected = exp_gp[pred_gp.columns[0]] > 0
    correct_detections = np.sum(pred_detected == actual_detected)
    total_values = len(pred_gp)
    accuracy = (correct_detections / total_values) * 100
    return accuracy

#Turn SHAPvalue for a single sample into a melt df that is ready to plot the network
#Input: SHAP output value for one sample, glycan names, lectin names, the threshold for which SHAP value to plot
#Output: a melt df with glycan and lectin and SHAP value as columns.
def melt_SHAPvalue_for_plot(shap_value_single,glycan_name,lectin_name,weight_plot_threshold):
    # reshape_shap_value = [item for sublist in shap_value_single for item in sublist]
    SHAP_value_df = pd.DataFrame(shap_value_single,index=glycan_name,columns=lectin_name)#.reshape(65,8))
    SHAP_value_df['glycan']=SHAP_value_df.index
    SHAP_value_melt=pd.melt(SHAP_value_df,id_vars=['glycan'],var_name='lectins',value_name='SHAPvalue',ignore_index=True)
    SHAP_value_melt.drop(labels=SHAP_value_melt.index[abs(SHAP_value_melt['SHAPvalue'])<weight_plot_threshold],inplace=True)#,axis=1)
    return SHAP_value_melt

#Plot SHAPvalue into a bipartite network. Shows every lectin nodes. Shows some glycan nodes using threshold.
#Edges represent SHAPvalue, not all edges are plotted.
#Input: melted SHAPvalue df, lectins in use, dictionary with lectin/glycan as key and their abundance as value
#Output: the plot...
def plot_SHAP(SHAPvalue_df,lectins,glycans,node_value_dict,figsizeDIY=(15,20),text_pos_factor=1,save_path=''):
    G = nx.from_pandas_edgelist(SHAPvalue_df.copy(), 'glycan', 'lectins', 'SHAPvalue')
    edgelist = [(u, v) for (u, v, d) in G.edges(data=True)]
    nodelist = set(u for edge in edgelist for u in edge)
    # To plot in bipartite style, assign each node to a part (0 for lectin or 1 for glycan)
    for node in G.nodes():
        G.nodes[node]['bipartite'] = 0 if node in lectins else 1
    #lectin nodes are the label_nodes, beacuse I only label their node names in plot
    label_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
    #Make sure every lectin is included, even they are not in the SHAP_value_melt
    if len(label_nodes) < len(lectins):
        missing_lectin = set(lectins) - set(label_nodes)
        for m in missing_lectin:
            G.add_node(m)
            G.nodes[m]['bipartite'] = 0
    #Update the node lists after updating missing lectins
    label_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
    # Define the desired order for your lectins; for example, alphabetical order:
    
    nodelist = nodelist.union(label_nodes) 
    glycan_nodes = nodelist - set(label_nodes)

    #Make sure every glycan is included, even they are not in the SHAP_value_melt
    if len(glycan_nodes) < len(glycans):
        missing_glycan = set(glycans) - set(glycan_nodes)
        for m in missing_glycan:
            G.add_node(m)
            G.nodes[m]['bipartite'] = 1
    glycan_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 1]
    nodelist = nodelist.union(glycan_nodes) 

    #To sort the nodes in the order desired
    label_nodes = sorted(label_nodes)
    glycan_nodes = sorted(list(glycan_nodes), key=lambda x: (len(x), x))
    y_positions = {}
    for idx, node in enumerate(label_nodes):
        y_positions[node] = -(idx / (len(label_nodes)-1))
    for idx, node in enumerate(glycan_nodes):
        y_positions[node] = -(idx / (len(glycan_nodes)-1))
    pos = nx.bipartite_layout(G, [n for n, d in G.nodes(data=True) if d['bipartite'] == 0])
    for node, coords in pos.items():
        pos[node] = (coords[0], y_positions[node])
#     print(pos)

    #Plotting
    fig, ax = plt.subplots(figsize=figsizeDIY)
    ax.set_xlim(-1.5, 1.5)  
    # create a custom colormap with blue for negative, white for zero and red for positive
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["blue","white","red"])
    sizes = [node_value_dict.get(node, 0)*300 for node in nodelist] #node size representing relative abun.
    
    #Draw the nodes
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist,node_size=sizes,node_color='black', ax=ax)

    #Draw the edges
    weights = nx.get_edge_attributes(G, 'SHAPvalue')
    max_abs_value = max(abs(min(weights.values())), abs(max(weights.values())))# get the maximum absolute SHAPvalue
    edge_colors = [cmap((weights[edge]+max_abs_value) / (2*max_abs_value)) for edge in G.edges()]# Normalize weights to range [-1, 1] and calculate the corresponding colors
    nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=3,edge_color=edge_colors, edge_cmap=plt.cm.Blues, ax=ax)#
    #Draw the labels for nodes in label_nodes(lectin) only
    labels = {node: node for node in label_nodes}
    label_pos = {node: (-1.2, coord[1]) for node, coord in pos.items() if node in label_nodes}
    print(label_pos)
    nx.draw_networkx_labels(G, label_pos, labels=labels,font_size=figsizeDIY[0]*1.33, ax=ax)

    # Function to add image on a node
    def add_image(image_path, xy, zoom, rotation, ax):
        img = mpimg.imread(image_path)
        img = img / img.max()  # normalize to [0,1]
        img_rotated = ndimage.rotate(img, rotation)
        img_rotated = np.clip(img_rotated, 0, 1)  # clip after rotation
        im = OffsetImage(img_rotated, zoom=zoom)
        ab = AnnotationBbox(im, xy, frameon=False)
        ax.add_artist(ab)

    # Draw colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-0.2, vmax=0.2)) 
    cbar = plt.colorbar(sm,ax=ax,fraction=0.02) #fraction control the thickness 
    cbar.set_label('SHAP value', rotation=270, labelpad=20)
    #Draw glycan icon images by their nodes
    pos_shifted = {node: (coords[0]*1.3, coords[1]) for node, coords in pos.items()}#pos of the glycans, shift glycan images to the right of the nodes

    for glycan in glycan_nodes:
        add_image("../Figure/Glycan_icon/"+glycan+".png", pos_shifted[glycan], zoom=0.2, rotation=0, ax=ax)
    #Add legend for the size of nodes to represents glycan abundances
    legend_dot_size=[i * 300 for i in [0,0.05,0.1,0.2,0.3,0.5]]
    legend_elements = [mlines.Line2D([0], [0], color='black', marker='o', markersize=size**0.5, label=str(size/3)+'%', linestyle='None') for size in legend_dot_size]
    plt.legend(bbox_to_anchor=(1.6, 0.3),handles=legend_elements, title='Relative abundance(%)')

    ax.axis('off')#remove black frame
    plt.tight_layout()
    plt.savefig(save_path+".png", dpi=300, bbox_inches='tight')
    plt.show()
    return glycan_nodes


#Sum the SHAP values for each lectin, and then plot them as bars to see who had the most impact
#Input: a melt df that have SHAP value for each lectin, each glycan
#Output: none, but will give a plot.
def barplot_sumSHAP_lectins(df,ylim_upper=1,save_path=''):
    # Separate positive and negative values
    df_positive = df[df['SHAPvalue'] >= 0]
    df_negative = df[df['SHAPvalue'] < 0]

    # Group by labels and calculate sums for both positive and negative
    sums_positive = df_positive.groupby('lectins')['SHAPvalue'].sum()
    sums_negative = np.abs(df_negative.groupby('lectins')['SHAPvalue'].sum())  # Take absolute values here

    sum_df = pd.concat([sums_positive, sums_negative], axis=1, keys=['SHAPvalue>=0', 'SHAPvalue<0'])
    sum_df = sum_df.fillna(0)
    # Calculate total absolute sum for each label
    sum_df['Total'] = sum_df['SHAPvalue>=0'] + sum_df['SHAPvalue<0']
    # Sort the DataFrame by the total sum in descending order
    sum_df = sum_df.sort_values(by='Total', ascending=False)
    sum_df = sum_df.drop(columns=['Total'])

    sum_df.plot(kind='bar', stacked=True, color=['red', 'blue'], rot=45, ylim=(0,ylim_upper), figsize=(6, 6))
    plt.gcf().savefig(save_path+".png", dpi=300, bbox_inches='tight')
    plt.show()

    return