"""
.py file originally from the NEST microcrocircuit model, modified to include analysis of the connectivity and synchrony of the network.
"""


import os
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import scipy as sp 
import numpy as np
import pandas as pd
import nest 
import helpers

from scipy.sparse import dok_matrix 
from scipy.fft import fft
from scipy.fft import fftfreq
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import scienceplots
plt.style.use(['science'])


analysis_dict = {
    "analysis_start": 500, #900  
    "analysis_end": 5500, #1100 #60500

    "start_periodic": 950,
    "end_periodic": 1450,

    "name": "connectivity_alter_no_stimulus/", 
    "synchrony_start": 500, #1000
    "synchrony_end": 5500, #1100 #60500

    
    "convolve_bin_size": 0.2,
    "bin_size": 3,
    }

def number_synapses(net_pops,number = 50):
    '''Computes de number of synapses.

    For a number of target cells, it calculates the number of synapses that are targetting those.
    The number is divided in excitatory and inhibitory synapses.

    Parameters
    ----------
    net_pops (nest network NodeCollection)
        a NodeCollection containing the information of all populations
    number (int)
        the number of neurons to measure the synapses for each population

    Returns
    -------
        data_synapses
            a dictionary containing the dataframes with the information of the number of neurons targetting each cell.
            each index of the dictionary is a population' dataframe.
            the dataframe contains number columns, and two rows, one for incoming excitatory synapses, and one for incoming inhibitory synapses
    
    '''
    data_synapses = {}
    j = 0

    population_limits = np.loadtxt("data_og/population_nodeids.dat")
    ex_populations = np.array(population_limits[::2],dtype=int)
    ranges = []
    mean_ex = []
    std_ex = []
    mean_in = []
    std_in= []
    for i in range(len(ex_populations)):
        ranges =  np.append(ranges,range(ex_populations[i][0],ex_populations[i][1]))

    for pop in net_pops:
        column_list = []
        counter = 0
        data_synapses[j] = {}
        data_frame = {}
        print("\r pop: %d " %(j), end = '\n',flush = True)
        for neuron in pop:
            connections = nest.GetConnections(target=neuron)
            conn_vals = connections.get(["target","source"])
            column_list = np.append(column_list,conn_vals["target"][0])
            print("\r neuron: %d " % (conn_vals["target"][0]),end = '',flush=True)
            ex_counts = 0
            in_counts = 0
            for i in range(len(conn_vals["source"])):
                if conn_vals["source"][i] in ranges: # I think this counting process could be optimised, but that is currently on standby
                    ex_counts = ex_counts +1
                else:
                    in_counts = in_counts + 1
            data_frame[conn_vals["target"][0]] = [ex_counts,in_counts]
            counter = counter +1 
            if counter >= number:
                break
        help = np.array(list(data_frame.values()))
        mean_ex = np.append(mean_ex,np.mean(help[:,0]))
        mean_in = np.append(mean_in,np.mean(help[:,1]))
        std_ex = np.append(std_ex,np.sqrt(np.var(help[:,0])))
        std_in = np.append(std_in,np.sqrt(np.var(help[:,1])))
        dataframe = pd.DataFrame(data=data_frame, columns= column_list, index=["excitatory","inhibitory"])
        print("Mean:" +str(dataframe.values.mean(axis=1)) + 'Standard deviation:'+str(dataframe.values.std(axis=1))+'Maximum:'+str(dataframe.values.max(axis=1))+'Minimum:'+str(dataframe.values.min(axis=1)))
        dataframe.to_csv("synapses_data/pop"+ str(j)+".txt")
        data_synapses[j] = dataframe
        j = j +1
    
    np.savetxt("synapses_data/mean_ex.dat",mean_ex)
    np.savetxt("synapses_data/std_ex.dat",std_ex)
    np.savetxt("synapses_data/mean_in.dat",mean_in)
    np.savetxt("synapses_data/std_in.dat",std_in)

    return data_synapses

def gather_metadata(path, name):
    """Reads names and ids of spike recorders and first and last ids of
    neurons in each population.

    If the simulation was run on several threads or MPI-processes, one name per
    spike recorder per MPI-process/thread is extracted.

    Parameters
    ------------
    path
        Path where the spike recorder files are stored.
    name
        Name of the spike recorder, typically ``spike_recorder``.

    Returns
    -------
    sd_files
        Names of all files written by spike recorders.
    sd_names
        Names of all spike recorders.
    node_ids
        Lowest and highest id of nodes in each population.

    """
    # load filenames
    sd_files = []
    sd_names = []
    for fn in sorted(os.listdir(path)):
        if fn.startswith(name):
            sd_files.append(fn)
            fnsplit = "-".join(fn.split("-")[:-1])
            if fnsplit not in sd_names:
                sd_names.append(fnsplit)
    
    #load node IDs
    node_idfile = open(path + "population_nodeids.dat","r")
    node_ids = []
    for node_id in node_idfile:
        node_ids.append(node_id.split())
    node_ids = np.array(node_ids, dtype="i4")
    return sd_files, sd_names, node_ids

def load_data(path, name,type="Voltage"):
    """ Reads all the voltage data recorded during the simulation.

    Parameters
    ----------
    path (string)
        the location of the folder which contains all of the recorded data
    name (string)
        the specific naming convention for all of the datafiles, usually selected as a simulation parameter
    type (string, either 'Voltage', 'Current')
        the type of data to be read

    Returns
    -------
    None
        Only if type is not either of the two options
    data (dict)
        A dictionary containign the read data, each index corresponds to a single population
    """
    sd_files, sd_names, node_ids = gather_metadata(path,name)
    data = {}
    if type=="Voltage":
        dtype = {"names": ("sender","time_ms","V_m"), "formats": ("i4","f8","f8")}
    elif type=="Current":
        dtype = {"names": ("sender","time_ms","I_syn"), "formats": ("i4","f8","f8")}
    else:
        print("type must be either Voltage or Current")
        return 

    for i,name in enumerate(sd_names):
        data_i_raw = np.array([[]], dtype=dtype)
        for j, f in enumerate(sd_files):
            if name in f:
                ld = np.loadtxt(os.path.join(path,f), skiprows=3,dtype=dtype)
                data_i_raw = np.append(data_i_raw, ld)

        data_i_raw = np.sort(data_i_raw, order = "time_ms")
        data[i] = data_i_raw

    return data

def split_data(data,num_neurons,type="Voltage"):
    """ Reads all the voltage data recorded during the simulation.

    Parameters
    ----------
    data (array)
        the data for a population
    num_neurons (int)
        the number of neurons in that population
    type (string, either 'Voltage', 'Current')
        the type of data to be read

    Returns
    -------
    None
        Only if type is not either of the two options
    data (dict)
        A dictionary containign the read data, each index corresponds to a single population
    """
    data_pop = np.zeros((num_neurons,len(data["time_ms"][0::num_neurons])))
    for i in range(num_neurons):
        if type=="Voltage":
            data_pop[i][:] = data["V_m"][i::num_neurons]
        if type=="Current":
            data_pop[i][:] = data["I_syn"][i::num_neurons]
    return data_pop

def get_time(data,num_neurons):
    """I think this function can be removed, but i'll wait to check
    """
    return data[0]["time_ms"][0::num_neurons]


def analyse_synchrony(num_neurons,bin_width=3,t_r = 2,dt=0.01):
    analysis_interval_start = analysis_dict["analysis_start"]
    analysis_interval_end = analysis_dict["analysis_end"]
    analysis_interval_start_s = analysis_dict["synchrony_start"]
    analysis_interval_end_s = analysis_dict["synchrony_end"]
    
    
    analysis_length = analysis_interval_end - analysis_interval_start
    analysis_length_s = analysis_interval_end_s - analysis_interval_start_s
    pop_activity = {}
    if analysis_length - analysis_length_s < 0:
        print('There is a problem. Synchrony measurement range must be smaller than the other')
        return 

    sd_names, node_ids, data = helpers.__load_spike_times("data_og/","spike_recorder",analysis_interval_start_s, analysis_interval_end_s)
    
    helper = np.loadtxt(analysis_dict["name"]+"pop_activities/pop_activity_"+str(0)+".dat")
    sum_array = np.zeros_like(helper)
    for i in range(len(num_neurons)):
        pop_activity[i] = np.loadtxt(analysis_dict["name"]+"pop_activities/pop_activity_"+str(i)+".dat")
        sum_array = sum_array + pop_activity[i]
    
    data_s = {}

    for i in data:
        low = np.searchsorted(data[i]["time_ms"],v=analysis_interval_start_s,side="left")
        high = np.searchsorted(data[i]["time_ms"],v=analysis_interval_end_s,side='right')
        data_s[i] = data[i][low:high]


    synchrony_pd = []
    synchrony_chi = []
    irregularity = []
    irregularity_pdf = {}

    super_lvr = []
    lvr_pdf = {}

    times_s = {}

    for i, n in enumerate(sd_names):

        #Computing synchrony
        neurons = np.unique(data_s[i]["sender"])
        random.shuffle(neurons)
        chosen_ones = neurons[1:1000]
        indices = []
        for indx in chosen_ones:
            indices = np.append(indices,np.where(data_s[i]["sender"]==indx))
        indices = np.array(indices,dtype=int)
        times_s[i] = data_s[i][indices]["time_ms"]
        counts, bins = np.histogram(times_s[i], bins=int(analysis_length_s/bin_width))
        counts2, bins = np.histogram(times_s[i], bins=int(analysis_length_s/bin_width/2))
    
        synchrony_chi = np.append(synchrony_chi,np.var(counts2)/np.mean(counts2)) 
        synchrony_pd = np.append(synchrony_pd,np.var(counts)/np.mean(counts))

        #Computing irregularity and LvR
        single_irregularity = []
        used_senders = []
        single_lvr = 0
        lvr = []

    mean_pop = sum_array / len(num_neurons)
    sum = 0
    for i, n in enumerate(pop_activity):
        sum = sum + np.var(pop_activity[i])
    chi = np.var(mean_pop) / ((1 / len(num_neurons)) * sum)

    
    for i,n in enumerate(sd_names):
        single_irregularity = []
        used_senders = []
        single_lvr = 0
        lvr = []
        for senders in data[i]["sender"]:
            individual_neurons = []
            times = []
            isi = []
            count = 0
            if senders not in used_senders:
                used_senders = np.append(used_senders,senders)
                individual_neurons = np.append(individual_neurons,np.where(data[i]["sender"]==senders))
                for index in individual_neurons:
                    times = np.append(times,data[i][int(index)]["time_ms"])

                if len(times)>4:
                    for j in range(len(times)-1):
                        isi = np.append(isi,times[j+1]-times[j])
                    for j in range(len(isi)-1):
                        single_lvr = single_lvr +  (1 - 4*isi[j]*isi[j+1] / (isi[j]+isi[j+1])**2) * (1 + 4 * t_r / (isi[j] + isi[j+1]))

                    neuron_rate = len(times) / analysis_length * 1000

                    single_lvr = 3 / (len(isi)-1) * single_lvr
                    lvr = np.append(lvr, single_lvr)
                    mean =np.mean(isi)
                    var = np.sqrt(np.var(isi))
                    single_irregularity = np.append(single_irregularity,np.float128(var/mean))
                    count = count + 1

        super_lvr = np.append(super_lvr, np.mean(lvr))
        lvr_pdf[i] = lvr
        irregularity = np.append(irregularity,np.mean(single_irregularity))
        irregularity_pdf[i] = single_irregularity
        if count >= 1000:
            break

    super_lvr = super_lvr[~np.isnan(super_lvr)]
    irregularity = irregularity[~np.isnan(irregularity)] 

    return synchrony_pd, synchrony_chi, irregularity, irregularity_pdf, super_lvr, lvr_pdf, times_s, chi


def analyse_firing_rates():
    analysis_interval_start = analysis_dict["analysis_start"]
    analysis_interval_end = analysis_dict["analysis_end"]
    analysis_length = analysis_interval_end - analysis_interval_start
    
    sd_names, node_ids, data = helpers.__load_spike_times("data_og/","spike_recorder",analysis_interval_start, analysis_interval_end)

    spike_rates = {}
    super_times = []

    for i, n in enumerate(sd_names):
        single_spike_rates = []
        used_senders = []
        for senders in data[i]["sender"]:
            individual_neurons = []
            times = []
            isi = []
            count = 0
            if senders not in used_senders:
                used_senders = np.append(used_senders,senders)
                individual_neurons = np.append(individual_neurons,np.where(data[i]["sender"]==senders))
                for index in individual_neurons:
                    times = np.append(times,data[i][int(index)]["time_ms"])
                if len(times)>2:
                    neuron_rate = len(times) / analysis_length * 1000
                    single_spike_rates = np.append(single_spike_rates,neuron_rate)
                    count = count + 1
        spike_rates[i] = single_spike_rates
        if count >= 1000:
            break
    return spike_rates

#def prepare_data(data_pop,ex_current_pop,in_current_pop):
def prepare_data(data_pop):
    sd_names, node_ids, data = helpers.__load_spike_times("data_og/","spike_recorder",analysis_dict["analysis_start"], analysis_dict["analysis_end"])

    times = {}
    data_voltages = { }
    data_excitatory = {}
    data_inhibitory = {}
    bins = {}

    names = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]



    for i in range(len(data_pop)):
        random.shuffle(data_pop[i])
        #random.shuffle(ex_current_pop[i])
        #random.shuffle(in_current_pop[i])
        data_voltages[names[i]] =  np.mean(data_pop[i][0:1000],axis=0)
        #data_excitatory[names[i]] = np.mean(ex_current_pop[i][0:1000],axis=0)
        #data_inhibitory[names[i]] = np.mean(in_current_pop[i][0:1000],axis=0)
        neurons = np.unique(data[i]["sender"]) 
        random.shuffle(neurons)
        chosen_ones = neurons[1:1000]
        indices = []
        for indx in chosen_ones:
            indices = np.append(indices,np.where(data[i]["sender"]==indx))
        indices = np.array(indices,dtype=int)
        times_help = data[i][indices]["time_ms"] 
        times[names[i]], bins[names[i]] = np.histogram(data[i][indices]["time_ms"], bins = int(analysis_dict["analysis_end"]-analysis_dict["analysis_start"]/analysis_dict["bin_size"]))

    return data_voltages, times, times_help
    #return data_voltages, data_excitatory, data_inhibitory, times, times_help


def plot_correlations(data_voltages,data_excitatory,data_inhibitory,pop_activity,times,save_data=True):
    connection_p =np.array([[0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0, 0.0076, 0.0],
            [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0, 0.0042, 0.0],
            [0.0077, 0.0059, 0.0497, 0.135, 0.0067, 0.0003, 0.0453, 0.0],
            [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0, 0.1057, 0.0],
            [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0],
            [0.0548, 0.0269, 0.0257, 0.0022, 0.06, 0.3158, 0.0086, 0.0],
            [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
            [0.0364, 0.001, 0.0034, 0.0005, 0.0277, 0.008, 0.0658, 0.1443],
        ]
    )
    names = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
    dataframe = pd.DataFrame(data=data_voltages, columns= names)
    matrix = dataframe.corr(method='pearson')

    dataframe_ex = pd.DataFrame(data=data_excitatory, columns= names)
    matrix_ex = dataframe_ex.corr(method='pearson')

    dataframe_in = pd.DataFrame(data=data_inhibitory, columns= names)
    matrix_in = dataframe_in.corr(method='pearson')
    dataframe_activity = pd.DataFrame(data=pop_activity, columns= names)
    matrix_activity = dataframe_activity.corr(method='pearson')

    dataframe_times = pd.DataFrame(data=times, columns= names)
    matrix_times = dataframe_times.corr(method='pearson')

    variables = []
    for i in matrix.columns:
        variables.append(i)

    variablest = []
    for i in matrix_times.columns:
        variablest.append(i)

    variables_ex = []
    for i in matrix_ex.columns:
        variables_ex.append(i)

    variables_activity = []
    for i in matrix_activity.columns:
        variables_activity.append(i)

    variables_in = []
    for i in matrix_in.columns:
        variables_in.append(i)

    plt.figure(figsize=(25,10))

    # Adding labels to the matrix
    plt.subplot(2, 3, 1)
    plt.imshow(matrix, cmap='Blues')
    plt.colorbar()
    plt.title('Mean Voltage Correlation')
    plt.xticks(range(len(matrix)), variables, rotation=45, ha='right')
    plt.yticks(range(len(matrix)), variables)

    plt.subplot(2, 3, 2)    
    plt.imshow(matrix_ex, cmap='Blues')
    plt.colorbar()
    plt.title('Excitatory Current Correlation')
    plt.xticks(range(len(matrix_ex)), variables_ex, rotation=45, ha='right')
    plt.yticks(range(len(matrix_ex)), variables_ex)

    plt.subplot(2, 3, 3)
    plt.imshow(matrix_in, cmap='Blues')
    plt.colorbar()
    plt.title('Inhibitory Current Correlation')
    plt.xticks(range(len(matrix_in)), variables_in, rotation=45, ha='right')
    plt.yticks(range(len(matrix_in)), variables_in)

    plt.subplot(2, 3, 4)
    plt.imshow(matrix_activity, cmap='Blues')
    plt.colorbar()
    plt.title('Pop Activity Correlation')
    plt.xticks(range(len(matrix_activity)), variables_activity, rotation=45, ha='right')
    plt.yticks(range(len(matrix_activity)), variables_activity)    
    
    plt.subplot(2, 3, 5)
    plt.imshow(matrix_times, cmap='Blues')
    plt.colorbar()
    plt.title('Spike Trains Correlation')
    plt.xticks(range(len(matrix_times)), variablest, rotation=45, ha='right')
    plt.yticks(range(len(matrix_times)), variablest)

    plt.subplot(2, 3, 6)
    plt.imshow(connection_p, cmap = 'Greens')
    plt.xticks(ticks=[0,1,2,3,4,5,6,7],labels=["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"])
    plt.yticks(ticks=[0,1,2,3,4,5,6,7],labels=["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"])
    plt.title('Connection probabilities')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    if save_data:
        matrix.to_csv(analysis_dict["name"]+'voltage_correlation.dat', sep=' ')
        matrix_times.to_csv(analysis_dict["name"] + 'spiketimes_correlation.dat', sep=' ')
        matrix_ex.to_csv(analysis_dict["name"] + 'ex_current_correlation.dat', sep=' ')
        matrix_in.to_csv(analysis_dict["name"] + 'in_currents_correlation.dat', sep=' ')
        matrix_activity.to_csv(analysis_dict["name"] + 'activity_correlation.dat', sep=' ')

def plot_cross_correlation(signal_1,signal_packet,signal_name,time_lag = 50, corr_start = 0, corr_end = 200):
    fig = plt.figure(figsize=(20,11))
    ax = fig.add_subplot(121,label='1')
    ax2 = fig.add_subplot(122, label = "2")
    ax.plot(signal_1[corr_start:corr_end], color = "blue", label = "Original -"+signal_name)
    for i, n in enumerate(signal_packet):
        if str(n) != signal_name:
            corr = signal.correlate(signal_1[corr_start:corr_end],signal_packet[str(n)][corr_start-time_lag:corr_end+time_lag],mode='valid')
            lags = signal.correlation_lags(len(signal_1[corr_start:corr_end]), len(signal_packet[str(n)][corr_start-time_lag:corr_end+time_lag]),mode='valid')
            corr /= np.max(corr)
            ax.plot(signal_packet[str(n)][corr_start:corr_end],label = str(n))
            ax2.plot(lags, corr,label=str(n))

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.legend()
    ax.grid()
    ax2.set_xlabel("Discretised time lag")
    ax2.set_ylabel("Normalised correlation")
    ax2.grid()




def plot_synchrony(synchrony_pd, synchrony_chi, irregularity, irregularity_pdf, lvr, lvr_pdf,chi):
    plt.figure(figsize=(18,18))
    pops = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
    colours = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
    fs = 18
    medianprops = dict(linestyle="-", linewidth=2.5, color="black")
    meanprops = dict(linestyle="--", linewidth=2.5, color="darkgray")

    plt.subplot(3, 2, 1)

    plt.barh(pops[::-1], synchrony_pd[::-1], color = colours[::-1])
    #plt.ylabel('neuron populations', fontsize = fs)
    #plt.title('Synchrony')
    plt.xlabel('synchrony', fontsize = fs)
    plt.grid(alpha = 0.5)

    plt.subplot(3, 2, 2)
    plt.barh(pops[::-1], synchrony_chi[::-1], color = colours[::-1])
    #plt.ylabel('populations', fontsize = fs)
    plt.title(r'$\chi$ =('+str(round(chi,3))+')', fontsize = fs)
    plt.xlabel('synchrony with half bin size', fontsize = fs)
    plt.grid(alpha = 0.5)

    plt.subplot(3,2,3)
    #plt.barh(pops[::-1], irregularity[::-1], color = colours[::-1])
    test = []
    colours = ['#3288bd','#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5']
    label_pos = list(range(len(pops),0,-1))
    for i in np.arange(len(pops)):
        test.append(irregularity_pdf[i])
    bp = plt.boxplot(test, 0, "rs",0,medianprops=medianprops,meanprops=meanprops, meanline=True, showmeans=True, orientation='horizontal')
    #plt.ylabel('neuron populations', fontsize =fs)
    #plt.title('Irregularity')
    label_pos = list(range(len(pops), 0, -1))
    for i in np.arange(len(pops)):
        boxX = []
        boxY = []
        box = bp["boxes"][i]
        for j in list(range(5)):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        k = i % 2
        boxPolygon = Polygon(boxCoords, facecolor=colours[-i])
        plt.gca().add_patch(boxPolygon)
    plt.xlabel('irregulatiry', fontsize = fs)
    plt.yticks(label_pos, pops, fontsize=fs)
    plt.grid(alpha = 0.5)

    plt.subplot(3,2,4)
    colours = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']

    irregularity_total = []
    for i in irregularity_pdf:
        data, bins= np.histogram(irregularity_pdf[i],density=True,bins=50)
        irregularity_total = np.append(irregularity_total,irregularity_pdf[i])
        plt.plot(bins[:-1],data,alpha=0.3+i*0.1, label = pops[i], color = colours[i])

    data_t, bins_s = np.histogram(irregularity_total,density=True,bins=50)
    plt.plot(bins[:-1],data_t, label = 'Total', color = 'black', ls = 'dashed')
   
    plt.legend()
    plt.ylabel('P (irregularity)', fontsize = fs)
    #plt.title('P (irregularity)', fontsize = fs)
    plt.xlabel('CV ISI', fontsize = fs)
    plt.grid(alpha = 0.5)
    plt.xlim(0,1.8)
    plt.ylim(0,3.6)

    plt.subplot(3,2,5)
    colours = ['#3288bd','#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5']
    label_pos = list(range(len(pops),0,-1))
    test = []
    for i in np.arange(len(pops)):
        test.append(lvr_pdf[i])
    bp = plt.boxplot(test, 0, "rs",0,medianprops=medianprops,meanprops=meanprops, meanline=True, showmeans=True, orientation='horizontal')
    #plt.ylabel('neuron populations', fontsize =fs)
    #plt.title('Irregularity')
    label_pos = list(range(len(pops), 0, -1))
    for i in np.arange(len(pops)):
        boxX = []
        boxY = []
        box = bp["boxes"][i]
        for j in list(range(5)):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        k = i % 2
        boxPolygon = Polygon(boxCoords, facecolor=colours[-i])
        plt.gca().add_patch(boxPolygon)
    #plt.title('LvR')
    plt.yticks(label_pos, pops, fontsize=fs)
    plt.xlabel('LvR', fontsize = fs)
    plt.grid(alpha = 0.5)

    plt.subplot(3,2,6)
    colours = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
    lvr_total = []
    for i in lvr_pdf:
        data, bins= np.histogram(lvr_pdf[i],density=True)
        lvt_total = np.append(lvr_total,lvr_pdf[i])
        plt.plot(bins[:-1],data,alpha=0.3+i*0.1, label = pops[i], color = colours[i])

    data_t, bins_s = np.histogram(lvr_total,density=True)
    plt.plot(bins[:-1],data_t, label = 'Total', color = 'black', ls = 'dashed')
    plt.grid(alpha = 0.5)
    plt.legend()
    plt.ylabel('P (LvR)', fontsize = fs)
    #plt.title('P (LvR)')
    plt.xlim(0,8)
    plt.ylim(0,2.1)
    plt.xlabel('LvR', fontsize =fs)

def plot_firing_rates(spike_rates):
    plt.figure(figsize=(12,7))
    pops = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
    bar_labels = ['darkred', 'red', 'blue', 'aqua', 'green', 'lime', 'orange', 'moccasin']
    bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
    plt.subplot(1,2,1)
    for i in spike_rates:
        plt.hist(spike_rates[i],color=bar_labels[i],alpha = 0.3 + i*0.1, label = pops[i])
    plt.grid(alpha = 0.5)
    plt.legend()
    plt.ylabel('Counts')
    plt.title('Firing rate histogram')
    plt.xlabel('Firing rate (spikes/s)')

    plt.subplot(1,2,2)
    spike_total = []
    for i in spike_rates:
        data, bins= np.histogram(spike_rates[i],density=True)
        spike_total = np.append(spike_total,spike_rates[i])
        plt.plot(bins[:-1],data,alpha=0.3+i*0.1, label = pops[i], color = bar_labels[i])

    data_t, bins_s = np.histogram(spike_total,density=True)
    plt.plot(bins[:-1],data_t, label = 'Total', color = 'black', ls = 'dashed')
    plt.grid(alpha = 0.5)
    plt.legend()
    plt.ylabel('Normalised Counts')
    plt.title('P (rate)')
    plt.xlabel('Firing rate (spikes/s)')
    plt.savefig("test_synchrony.svg", dpi=300)
    plt.show()



def compute_FFT(signal_data,freq_sample= 0.001,freq_sample_welsh = 1000,lim_y = 7000, lim_x = 200, low_log = 10, high_log =90,fit=False,fit_freq_start = 4.0, fit_freq_end = 14.0,test_p0 = [30,10,5],welsh_fit = 'alpha',signal_xmin=500,signal_xmax=700,save=True,name_='freq.dat'):

    analysis_interval_start = analysis_dict["analysis_start"]
    analysis_interval_end = analysis_dict["analysis_end"]
    FFT_Results = {}
    Welsh_Freqs = {}
    Welsh_Powers = {}
    if fit:
        Fit_FFT = {}
        Fit_Welsh = {}
        mean_freq = []
        mean_welsh = []
        amplitude_freq = []
        amplitude_welsh = []
        sigma_freq = []
        sigma_welsh = []
        def gaus(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))


    for i in signal_data:
        FFT_Results[i] = fft(signal_data[i][analysis_interval_start:analysis_interval_end]-np.mean(signal_data[i][analysis_interval_start:analysis_interval_end]))
        Welsh_Freqs[i], Welsh_Powers[i]  = signal.welch(signal_data[i][analysis_interval_start:analysis_interval_end]-np.mean(signal_data[i][analysis_interval_start:analysis_interval_end]),fs=freq_sample_welsh)

    #Calcular los valores de frequencia correspondientes
    #freq = fftfreq(len(signal_data[i][analysis_interval_start:analysis_interval_end]),d=freq_sample) * 1000
    freq = fftfreq(len(signal_data[i][analysis_interval_start:analysis_interval_end]),d=freq_sample)
    index_start = int(np.where(freq==fit_freq_start)[0][0])
    index_end = int(np.where(freq==fit_freq_end)[0][0])


    plt.figure(figsize=(15, 5))
    colors = ['darkred', 'red', 'blue', 'aqua', 'green', 'lime', 'orange', 'moccasin']
    # Graficar la amplitud en función de la frecuencia

    plt.subplot(1, 4, 1)
    j= 0
    for i in signal_data:
        plt.plot(signal_data[i][analysis_interval_start:analysis_interval_end], c = colors[j], label = i)
        j=j+1
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage')
    plt.title('Signal')
    plt.xlim(signal_xmin,signal_xmax)    
    plt.grid(True)
    #plt.legend(loc= 'best')

    plt.subplot(1, 4, 2)
    j= 0

    indx = int(len(signal_data[i][analysis_interval_start:analysis_interval_end])/2)
    for i in FFT_Results:
        if fit:
            Fit_FFT[i], __ = curve_fit(gaus,freq[index_start:index_end],np.abs(FFT_Results[i][index_start:index_end]),p0 = test_p0)
            mean_freq = np.append(mean_freq,Fit_FFT[i][1])
            amplitude_freq = np.append(amplitude_freq,Fit_FFT[i][0])
            sigma_freq = np.append(sigma_freq,Fit_FFT[i][2])

            plt.plot(freq[index_start:index_end],gaus(freq[index_start:index_end],*Fit_FFT[i]),'--', c = colors[j])
        plt.plot(freq[:indx], np.abs(FFT_Results[i])[:indx],c = colors[j], label = i)
        j=j+1
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Voltage (minus the mean) FFT')
    plt.grid(True)
    plt.xlim(1,lim_x)
    plt.ylim(0,lim_y)
    plt.legend(loc= 'best')

    plt.subplot(1, 4, 3)
    j= 0
    indx = int(len(signal_data[i][analysis_interval_start:analysis_interval_end])/2)
    for i in FFT_Results:
        if fit:
            plt.plot(freq[index_start:index_end],20 * np.log10(gaus(freq[index_start:index_end],*Fit_FFT[i])),'--', c = colors[j])
        plt.plot(freq[:indx], 20 * np.log10(np.abs(FFT_Results[i])[:indx]),c = colors[j], label = i)
        j=j+1
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Voltage (minus the mean) FFT')
    plt.grid(True)
    plt.xlim(1,lim_x)
    plt.ylim(low_log,high_log)
    plt.legend(loc= 'best')

    plt.subplot(1, 4, 4)
    j= 0

    #plt.ylim([0.5e-3, 1])
    for i in Welsh_Freqs:
        if fit:
            if welsh_fit == 'alpha':
                i_start = int(np.where((Welsh_Freqs[i] < 4) & (Welsh_Freqs[i] > 0.0))[0][0])
                i_end = int(np.where((Welsh_Freqs[i]<22.0) & (Welsh_Freqs[i] >18.0))[0][0])
                p0 = [0.01,10,5]

            if welsh_fit == 'gamma':
                i_start = int(np.where((Welsh_Freqs[i] <60 ) & (Welsh_Freqs[i] > 50.0))[0][0])
                i_end = int(np.where((Welsh_Freqs[i]<120.0) & (Welsh_Freqs[i] >110.0))[0][0])
                p0 = [0.01,80,5]
        if fit:
            Fit_Welsh[i], __ = curve_fit(gaus,Welsh_Freqs[i][i_start:i_end],Welsh_Powers[i][i_start:i_end],p0 = p0)
            mean_welsh = np.append(mean_welsh,Fit_Welsh[i][1])
            amplitude_welsh = np.append(amplitude_welsh,Fit_Welsh[i][0])
            sigma_welsh = np.append(sigma_welsh,Fit_Welsh[i][2])
            plt.plot(Welsh_Freqs[i][i_start:i_end],gaus(Welsh_Freqs[i][i_start:i_end],*Fit_Welsh[i]),'--', c = colors[j])
        plt.plot(Welsh_Freqs[i], Welsh_Powers[i],c = colors[j], label = i)
        j=j+1
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'PSD $[V^2/Hz]$')
    plt.title('Voltage (minus the mean) Welch')
    plt.grid(True)
    #plt.legend(loc= 'best')
    plt.tight_layout()
    plt.xlim(1,lim_x)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

    if save:
        mean_final = np.mean( np.array([mean_freq,mean_welsh]),axis= 0)
        amplitude_final = np.mean( np.array([amplitude_freq,amplitude_welsh]),axis= 0)
        sigma_final = np.mean( np.array([sigma_freq,sigma_welsh]),axis= 0)
        pops = [0,1,2,3,4,5,6,7]
        np.savetxt(analysis_dict["name"] + name_, np.c_[pops,mean_final, amplitude_final, sigma_final], fmt = '%.2f', header = 'Pops mean_freq amplitude sigma')

    return FFT_Results, freq, Welsh_Freqs, Welsh_Powers, indx

def butter_bandpass(lowcut, highcut, fs, order=5,btype='band'):
    return butter(order, [lowcut, highcut], fs=fs, btype=btype)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5,btype='band'):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order,btype=btype)
    y = lfilter(b, a, data)
    return y

def filter_signal(data,fs,lowcut,highcut,order=3):
    filtered_signal = {}
    for i in data:
        filtered_signal[str(i)] = butter_bandpass_filter(data[str(i)][analysis_dict["analysis_start"]:analysis_dict["analysis_end"]]-np.mean(data[str(i)][analysis_dict["analysis_start"]:analysis_dict["analysis_end"]]),lowcut,highcut,fs,order)

    return filtered_signal

def plot_activity(pop_activity):
    plt.figure(figsize=(15, 5))
    pops = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
    bar_labels = ['darkred', 'red', 'blue', 'aqua', 'green', 'lime', 'orange', 'moccasin']
    times = np.linspace(analysis_dict["analysis_start"],analysis_dict["analysis_end"],int((analysis_dict["analysis_end"]-analysis_dict["analysis_start"])/analysis_dict["convolve_bin_size"]))
    mean_activity = []
    mean_maxima = []

    for i, n in enumerate(pop_activity):
        plt.plot(times,pop_activity[pops[i]],label=pops[i],color=bar_labels[i])
        mean_activity = np.append(mean_activity,np.mean(pop_activity[pops[i]]))
        peaks = find_peaks(pop_activity[pops[i]])
        mean_maxima = np.append(mean_maxima,np.mean(pop_activity[pops[i]][peaks[0]]))

    plt.xlabel('Time(ms)')
    plt.ylabel('Activity')
    plt.xlim(1000,1200)
    plt.legend()
    plt.grid()
    return mean_activity, mean_maxima