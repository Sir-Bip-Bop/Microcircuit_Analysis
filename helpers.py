"""
.py file originally from the NEST microcrocircuit model, modified to include analysis of the
connectivity and synchrony of the network.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy import signal
import addons 
from sklearn.preprocessing import normalize

if "DISPLAY" not in os.environ:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.pyplot.style.use('science')


def num_synapses_from_conn_probs(conn_probs, popsize1, popsize2):
    """Computes the total number of synapses between two populations from
    connection probabilities.

    Here it is irrelevant which population is source and which target.

    Parameters
    ----------
    conn_probs
        Matrix of connection probabilities.
    popsize1
        Size of first population.
    popsize2
        Size of second population.

    Returns
    -------
    num_synapses
        Matrix of synapse numbers.
    """
    prod = np.outer(popsize1, popsize2) #a matrix of size len(popsize1) x len(popsize2)
    num_synapses = np.log(1.0 - conn_probs) / np.log((prod - 1.0) / prod)
    return num_synapses


def postsynaptic_potential_to_current(C_m, tau_m, tau_syn):
    r"""Computes a factor to convert postsynaptic potentials to currents.

    The time course of the postsynaptic potential ``v`` is computed as
    :math: `v(t)=(i*h)(t)`
    with the exponential postsynaptic current
    :math:`i(t)=J\mathrm{e}^{-t/\tau_\mathrm{syn}}\Theta (t)`,
    the voltage impulse response
    :math:`h(t)=\frac{1}{\tau_\mathrm{m}}\mathrm{e}^{-t/\tau_\mathrm{m}}\Theta (t)`,
    and
    :math:`\Theta(t)=1` if :math:`t\geq 0` and zero otherwise.

    The ``PSP`` is considered as the maximum of ``v``, i.e., it is
    computed by setting the derivative of ``v(t)`` to zero.
    The expression for the time point at which ``v`` reaches its maximum
    can be found in Eq. 5 of [1]_.

    The amplitude of the postsynaptic current ``J`` corresponds to the
    synaptic weight ``PSC``.

    References
    ----------
    .. [1] Hanuschkin A, Kunkel S, Helias M, Morrison A and Diesmann M (2010)
           A general and efficient method for incorporating precise spike times
           in globally time-driven simulations.
           Front. Neuroinform. 4:113.
           DOI: `10.3389/fninf.2010.00113 <https://doi.org/10.3389/fninf.2010.00113>`__.

    Parameters
    ----------
    C_m
        Membrane capacitance (in pF).
    tau_m
        Membrane time constant (in ms).
    tau_syn
        Synaptic time constant (in ms).

    Returns
    -------
    PSC_over_PSP
        Conversion factor to be multiplied to a `PSP` (in mV) to obtain a `PSC`
        (in pA).

    """
    sub = 1.0 / (tau_syn - tau_m)
    pre = tau_m * tau_syn / C_m * sub
    frac = (tau_m / tau_syn) ** sub

    PSC_over_PSP = 1.0 / (pre * (frac**tau_m - frac**tau_syn))
    return PSC_over_PSP


def dc_input_compensating_poisson(bg_rate, K_ext, tau_syn, PSC_ext):
    """Computes DC input if no Poisson input is provided to the microcircuit.

    Parameters
    ----------
    bg_rate
        Rate of external Poisson generators (in spikes/s).
    K_ext
        External indegrees.
    tau_syn
        Synaptic time constant (in ms).
    PSC_ext
        Weight of external connections (in pA).

    Returns
    -------
    DC
        DC input (in pA) which compensates lacking Poisson input.
    """
    DC = bg_rate * K_ext * PSC_ext * tau_syn * 0.001
    return DC


def adjust_weights_and_input_to_synapse_scaling(
    full_num_neurons,
    full_num_synapses,
    K_scaling,
    mean_PSC_matrix,
    PSC_ext,
    tau_syn,
    full_mean_rates,
    DC_amp,
    poisson_input,
    bg_rate,
    K_ext,
):
    """Adjusts weights and external input to scaling of indegrees.

    The recurrent and external weights are adjusted to the scaling
    of the indegrees. Extra DC input is added to compensate for the
    scaling in order to preserve the mean and variance of the input.

    Parameters
    ----------
    full_num_neurons
        Total numbers of neurons.
    full_num_synapses
        Total numbers of synapses.
    K_scaling
        Scaling factor for indegrees.
    mean_PSC_matrix
        Weight matrix (in pA).
    PSC_ext
        External weight (in pA).
    tau_syn
        Synaptic time constant (in ms).
    full_mean_rates
        Firing rates of the full network (in spikes/s).
    DC_amp
        DC input current (in pA).
    poisson_input
        True if Poisson input is used.
    bg_rate
        Firing rate of Poisson generators (in spikes/s).
    K_ext
        External indegrees.

    Returns
    -------
    PSC_matrix_new
        Adjusted weight matrix (in pA).
    PSC_ext_new
        Adjusted external weight (in pA).
    DC_amp_new
        Adjusted DC input (in pA).

    """
    PSC_matrix_new = mean_PSC_matrix / np.sqrt(K_scaling)
    PSC_ext_new = PSC_ext / np.sqrt(K_scaling)

    # recurrent input of full network
    indegree_matrix = full_num_synapses / full_num_neurons[:, np.newaxis]
    input_rec = np.sum(mean_PSC_matrix * indegree_matrix * full_mean_rates, axis=1)

    DC_amp_new = DC_amp + 0.001 * tau_syn * (1.0 - np.sqrt(K_scaling)) * input_rec

    if poisson_input:
        input_ext = PSC_ext * K_ext * bg_rate
        DC_amp_new += 0.001 * tau_syn * (1.0 - np.sqrt(K_scaling)) * input_ext
    return PSC_matrix_new, PSC_ext_new, DC_amp_new


# Saturated colours for the poster panels: (raster dots, population-rate trace).
# L2/3E stays blue and L4E red-family, matching the layer colours used in the
# spectral entropy, autocorrelation and AIS figures.
POSTER_COLOURS = {
    0: ("#1f4fe0", "#08306b"),   # L2/3E: saturated blue dots, dark blue trace
    2: ("#e31a1c", "#99000d"),   # L4E:   saturated red dots,  dark red trace
}


def plot_raster(path, name, begin, end, N_scaling,binned,M, std,trial,plot,layer=0,rate=8.0,num_neurons = [20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948],
                dot_color=None, line_color=None, dot_alpha=None, line_alpha=None,
                markersize=None, save_suffix=""):
    """Creates a spike raster plot of the network activity.

    Parameters
    -----------
    path
        Path where the spike times are stored.
    name
        Name of the spike recorder.
    begin
        Time point (in ms) to start plotting spikes (included).
    end
        Time point (in ms) to stop plotting spikes (included).
    N_scaling
        Scaling factor for number of neurons.
    dot_color, line_color
        Override the raster-dot and population-rate colours in the single-
        population view (``plot=True``). ``None`` keeps the original pastel
        palette, so existing calls are unaffected.
    dot_alpha, line_alpha
        Override the transparencies (original: 0.4 for both). Saturated
        colours read better with more opaque dots and a fully opaque trace.
    markersize
        Raster dot size; ``None`` keeps the matplotlib default.
    save_suffix
        Appended to the saved filename so a restyled figure does not overwrite
        the original.

    Returns
    -------
    None

    """
    fs = 30  # fontsize
    ylabels = ["L2/3E", "L2/3I", "L4E", "L4I", "L5E", "L5I", "L6E","L6I"]
    color_list = ['#d6949c', '#f5b7a4', '#fcd4ac', '#ffeebf', '#edf5c9', '#d2ded1', '#b4c2be', '#7da4bd']
    bar_labels =  ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
    

    sd_names, node_ids, data = __load_spike_times(path, name, begin, end)
    last_node_id = node_ids[-1, -1]
    mod_node_ids = np.abs(node_ids - last_node_id) + 1
    neuron_pos = np.zeros(8)
    for i in np.arange(0,8,1):
        neuron_pos[i] = np.abs(node_ids[i,1] - node_ids[-1,1]) +1
    label_pos = np.zeros(8)
    for i in np.arange(0,8,1):
        if i%2 == 0:
            label_pos[i] = (mod_node_ids[i, 0] + mod_node_ids[i + 1, 1]) / 2.0
    filtered_signal = {}
    
    
    if binned:
        stp = 1
        if N_scaling > 0.1 and N_scaling <= 1:
            stp = int(10.0 * N_scaling)
            print("  Only spikes of neurons in steps of {} are shown.".format(stp))

        if trial ==0 :
            if plot:
                fig = plt.figure(figsize=(20,8),dpi=400)
            else:
                fig = plt.figure(figsize=(11,16))
            ax = fig.add_subplot(111,label='1')
            ax2 = fig.add_subplot(111, label = "2", frame_on=False)

        for i, n in enumerate(sd_names):
            times = data[i]["time_ms"]
            times_currents = np.linspace(begin,end,num=int((end-begin)/0.2))
            neurons = np.abs(data[i]["sender"] - last_node_id) + 1
            pop_activity, bins = np.histogram(times,bins=int((end-begin)/addons.analysis_dict["convolve_bin_size"]))
            window = signal.windows.gaussian(M[i],std[i])
            filtered_signal[i] = signal.convolve(pop_activity,window,mode='same')
            #filtered_signal[i] = signal.convolve(pop_activity,np.ones(5)/5,mode='same')
            lowcut_gamma = 25 #35
            highcut_gamma = 40 #95
            #filtered_signal[i] =  addons.butter_bandpass_filter(filtered_signal[i],lowcut= lowcut_gamma,highcut=highcut_gamma,fs=1000,order=3)
            if plot:
                norm = 1
            else:
                norm = np.linalg.norm(filtered_signal[i])
            #norm = np.linalg.norm(filtered_signal[i]) / np.linalg.norm(filtered_signal[i])
            try:
                high = neurons[-1]
            except  IndexError:
                high = 0
            try:
                low = neurons[0]
            except IndexError:
                low = 0
            if low == high:
                filtered_signal_plot = 0
            else:
                if plot:
                    filtered_signal_plot = filtered_signal[i] / norm / num_neurons[i] #* 3 * np.abs(high - low) + high
                else: 
                    filtered_signal_plot = filtered_signal[i] / norm * 5 * np.abs(high - low) + high
                if i%2!=0:
                    label_pos[i] = filtered_signal_plot[0]
                else:
                    if plot:
                        filtered_signal_plot = filtered_signal[i] /norm / num_neurons[i] #* 3 * np.abs(high - low) #+ label_pos[i]
                    else:
                        filtered_signal_plot = filtered_signal[i] /norm * 5 * np.abs(high - low) + label_pos[i]
            if trial ==0:
                if plot:
                    if i == layer:
                       #ax.plot(times[::stp], neurons[::stp], ".", color="#73a6db",alpha = 0.2)
                       #ax2.plot(times_currents,filtered_signal_plot, linewidth= 3, color="#3081d8",alpha= 0.4)
                       #ax.plot(times[::stp], neurons[::stp], ".", color="#96b6da",alpha = 0.2)
                       #ax2.plot(times_currents,filtered_signal_plot, linewidth= 3, color="#629ddb",alpha= 0.4)
                       #ax.plot(times[::stp], neurons[::stp], ".", color='gray',alpha = 0.6)
                       #ax2.plot(times_currents,filtered_signal_plot, linewidth= 3, color="black",alpha= 0.4)
                       dc = color_list[i] if dot_color is None else dot_color
                       lc = bar_labels[i] if line_color is None else line_color
                       da = 0.4 if dot_alpha is None else dot_alpha
                       la = 0.4 if line_alpha is None else line_alpha
                       mk = {} if markersize is None else {"markersize": markersize}
                       ax.plot(times[::stp], neurons[::stp], ".", color=dc, alpha=da, **mk)
                       ax2.plot(times_currents, filtered_signal_plot, linewidth=3, color=lc, alpha=la)
                       ax.tick_params(axis='x',labelsize=15)
                       ax.tick_params( axis='y',labelsize=15)
                       
                       #try:
                        #ax2.plot(times_currents,filtered_signal_plot, linewidth= 3, color=bar_labels[i])
                       #except ValueError:
                        #print("ValueError: times_currents and filtered_signal_plot have different lengths. Check the input data.")                        
                else:
                    ax.plot(times[::stp], neurons[::stp], ".", color=color_list[i],alpha = 0.3)
                    ax.axhline(y=neuron_pos[i], color= 'black', linestyle='--', alpha=1)
                    try:
                        ax2.plot(times_currents,filtered_signal_plot, linewidth= 3, color=bar_labels[i])
                    except ValueError:
                        print("ValueError: times_currents and filtered_signal_plot have different lengths. Check the input data.")


        if trial==0:
            if plot:
                test = []
                test = np.append(test, label_pos[2])
                test2 = []
                test2 = np.append(test2, ylabels[2])
                ax.set_xlabel("time [ms]", fontsize=fs)
                ax.set_ylabel("neurons", fontsize=fs)
                plt.title("Population " + ylabels[layer] + ", background rate=" + str(rate) + " spikes/s", fontsize=fs)
                ax.set_yticks([])
                ax2.set_xticks([])
                ax2.yaxis.tick_right()
                ax2.tick_params(axis='y', labelsize=20)
                ax.tick_params(axis='x', labelsize=20)
                ax2.ticklabel_format(axis='y', style='sci')
                ax2.yaxis.set_label_position("right")
                ax.set_xlim(begin,end)
                ax.set_ylim()
                ax2.set_xlim(begin,end)
                #ax.set_ylim(0,last_node_id)
                ax2.set_ylim(0,0.3)
                plt.savefig(os.path.join("Figure1/"+str(rate)+"raster_plot_"+str(layer)+save_suffix+".svg"), dpi=500)
            else:
                ax.set_xlabel("time (ms)", fontsize=fs)
                ax.set_yticks(label_pos, ylabels, fontsize=fs)
                ax2.set_xticks([])
                ax.set_xlim(begin,end)
                ax2.set_yticks([])
                ax2.set_xlim(begin,end)
                ax.set_ylim(0,last_node_id)
                ax2.set_ylim(0,last_node_id)
                plt.savefig(os.path.join("Figure1/"+str(rate)+"raster_plot_"+str(layer)+".svg"), dpi=500)
    else:
        color_list = bar_labels
        ylabels = ["L2/3","L4","L5","L6"]
        label_pos = [(mod_node_ids[i, 0] + mod_node_ids[i + 1, 1]) / 2.0 for i in np.arange(0, 8, 2)]
        stp = 1
        if N_scaling > 0.1:
            stp = int(10.0 * N_scaling)
            print("  Only spikes of neurons in steps of {} are shown.".format(stp))

        fig = plt.figure(figsize=(11,16))
        for i, n in enumerate(sd_names):
            times = data[i]["time_ms"]
            neurons = np.abs(data[i]["sender"] - last_node_id) + 1
            plt.plot(times[::stp], neurons[::stp], ".", color=color_list[i],alpha = 0.5)
        plt.xlabel("time (ms)", fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.xlim(begin,end)
        plt.ylim(0,last_node_id)
        plt.yticks(label_pos, ylabels, fontsize=fs)
        plt.savefig(os.path.join(path, "raster_plot.svg"), dpi=300)
    
    filtered_signal_complete = {}
    times_a = {}
    sd_names, node_ids, data_analysis = __load_spike_times(path, name, addons.analysis_dict["analysis_start"], addons.analysis_dict["analysis_end"])

    if os.path.isdir(os.path.join(path,"measurements/")):
        print("Directory already existed")
    else:
        os.mkdir(os.path.join(path,"measurements/"))

    if os.path.isdir(os.path.join(path,"measurements/pop_activities/")):
        print("Directory already existed")
    else:
        os.mkdir(os.path.join(path,"measurements/pop_activities/"))

    if os.path.isdir(os.path.join(path,"measurements/pop_activities2/")):
        print("Directory already existed")
    else:
        os.mkdir(os.path.join(path,"measurements/pop_activities2/"))


    for i, n in enumerate(sd_names):
        times_ = data_analysis[i]["time_ms"]
        pop_activity_a, times_a[i] = np.histogram(times_,bins=int((addons.analysis_dict["analysis_end"]-addons.analysis_dict["analysis_start"])/addons.analysis_dict["convolve_bin_size"]))
        window = signal.windows.gaussian(M[i],std[i])
        filtered_signal_complete[i] = signal.convolve(pop_activity_a,window,mode='same')
        np.savetxt(path +"measurements/pop_activities2/pop_activity_"+str(i)+".dat",filtered_signal_complete[i]/num_neurons[i])
        filtered_signal_complete[i] = (filtered_signal_complete[i] - np.min(filtered_signal_complete[i])) / (np.max(filtered_signal_complete[i])-np.min(filtered_signal_complete[i]))
        
        np.savetxt(path +"measurements/pop_activities/pop_activity_"+str(i)+".dat",filtered_signal_complete[i])

        if not os.path.exists(path + "measurements/times/"):
            os.makedirs(path + "measurements/times/")
        np.savetxt(path +"measurements/times/times_"+str(i)+".dat",times_a[i])

    return filtered_signal_complete, times_a
   

def firing_rates(path, name, begin, end):
    """Computes mean and standard deviation of firing rates per population.

    The firing rate of each neuron in each population is computed and stored
    in a .dat file in the directory of the spike recorders. The mean firing
    rate and its standard deviation are printed out for each population.

    Parameters
    -----------
    path
        Path where the spike times are stored.
    name
        Name of the spike recorder.
    begin
        Time point (in ms) to start calculating the firing rates (included).
    end
        Time point (in ms) to stop calculating the firing rates (included).

    Returns
    -------
    None

    """
    sd_names, node_ids, data = __load_spike_times(path, name, begin, end)
    all_mean_rates = []
    all_std_rates = []
    for i, n in enumerate(sd_names):
        senders = data[i]["sender"]
        # 1 more bin than node ids per population
        bins = np.arange(node_ids[i, 0], node_ids[i, 1] + 2)
        spike_count_per_neuron, _ = np.histogram(senders, bins=bins)
        rate_per_neuron = spike_count_per_neuron * 1000.0 / (end - begin)
        if os.path.isdir(os.path.join(path, ("rate" + str(i) + ".dat"))):
            os.remove(os.path.join(path, ("rate" + str(i) + ".dat")))
        np.savetxt(os.path.join(path, ("rate" + str(i) + ".dat")), rate_per_neuron)
        # zeros are included
        all_mean_rates.append(np.mean(rate_per_neuron))
        all_std_rates.append(np.std(rate_per_neuron))

    np.savetxt(os.path.join(path, ("measurements/mean_rate.dat")),all_mean_rates)
    np.savetxt(os.path.join(path, ("measurements/std_rate.dat")),all_std_rates)
    print("Mean rates: {} spikes/s".format(np.around(all_mean_rates, decimals=3)))
    print("Standard deviation of rates: {} spikes/s".format(np.around(all_std_rates, decimals=3)))


def boxplot(path, populations,trial):
    """Creates a boxblot of the firing rates of all populations.

    To create the boxplot, the firing rates of each neuron in each population
    need to be computed with the function ``firing_rate()``.

    Parameters
    -----------
    path
        Path where the firing rates are stored.
    populations
        Names of neuronal populations.

    Returns
    -------
    None

    """
    fs = 18
    pop_names = [string.replace("23", "2/3") for string in populations]
    label_pos = list(range(len(populations), 0, -1))
    color_list = ['#3288bd','#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5']
    reversed(color_list)
    medianprops = dict(linestyle="-", linewidth=2.5, color="black")
    meanprops = dict(linestyle="--", linewidth=2.5, color="darkgray")

    rates_per_neuron_rev = []
    for i in np.arange(len(populations))[::-1]:
        rates_per_neuron_rev.append(np.loadtxt(os.path.join(path, ("rate" + str(i) + ".dat"))))
    if trial == 0:
        plt.figure(figsize=(8, 6))
        bp = plt.boxplot(
            rates_per_neuron_rev, 0, "rs", 0, medianprops=medianprops, meanprops=meanprops, meanline=True, showmeans=True, orientation='horizontal'
        )
        plt.setp(bp["boxes"], color="black")
        plt.setp(bp["whiskers"], color="black")
        plt.setp(bp["fliers"], color="red", marker="+")
    
    
    # boxcolors
        for i in np.arange(len(populations)):
            boxX = []
            boxY = []
            box = bp["boxes"][i]
            for j in list(range(5)):
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])
            boxCoords = list(zip(boxX, boxY))
            k = i % 2
            boxPolygon = Polygon(boxCoords, facecolor=color_list[-i])
            plt.gca().add_patch(boxPolygon)
        plt.xlabel("firing rate (spikes/s)", fontsize=fs)
        plt.yticks(label_pos, pop_names, fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.grid()
        plt.savefig(os.path.join(path, "box_plot.svg"), dpi=300)



def __gather_metadata(path, name):
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
            # spike recorder name and its ID
            fnsplit = "-".join(fn.split("-")[:-1])
            if fnsplit not in sd_names:
                sd_names.append(fnsplit)

    # load node IDs
    node_idfile = open(path + "population_nodeids.dat", "r")
    node_ids = []
    for node_id in node_idfile:
        node_ids.append(node_id.split())
    node_ids = np.array(node_ids, dtype="i4")
    return sd_files, sd_names, node_ids


def __load_spike_times(path, name, begin, end):
    """Loads spike times of each spike recorder.

    Parameters
    ----------
    path
        Path where the files with the spike times are stored.
    name
        Name of the spike recorder.
    begin
        Time point (in ms) to start loading spike times (included).
    end
        Time point (in ms) to stop loading spike times (included).

    Returns
    -------
    data
        Dictionary containing spike times in the interval from ``begin``
        to ``end``.

    """
    sd_files, sd_names, node_ids = __gather_metadata(path, name)
    data = {}
    dtype = {"names": ("sender", "time_ms"), "formats": ("i4", "f8")}  # as in header
    for i, name in enumerate(sd_names):
        data_i_raw = np.array([[]], dtype=dtype)
        for j, f in enumerate(sd_files):
            if name in f:
                # skip header while loading
                ld = np.loadtxt(os.path.join(path, f), skiprows=3, dtype=dtype)
                data_i_raw = np.append(data_i_raw, ld)

        data_i_raw = np.sort(data_i_raw, order="time_ms")
        # begin and end are included if they exist
        low = np.searchsorted(data_i_raw["time_ms"], v=begin, side="left")
        high = np.searchsorted(data_i_raw["time_ms"], v=end, side="right")
        data[i] = data_i_raw[low:high]
    return sd_names, node_ids, data