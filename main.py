#!/usr/bin/env python
# coding: utf-8

# ### Preprocess the data

# In[1]:


import numpy as np
import h5py
from scipy.signal import butter, filtfilt, iirnotch
    
    
# Helper function to filter the data
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/ nyq, highcut/ nyq], btype='band', analog=False)
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def notch(notch_freq, samp_freq, quality_factor=30):
    b, a = iirnotch(notch_freq, quality_factor, samp_freq)
    return b, a

def notch_filter(data, notch_fs, fs, q=30):
    b, a = notch(notch_fs, fs, q)
    y = filtfilt(b, a, data)
    return y

def filt_GRID(data, lowcut=20, highcut=500, fs=4000, order=3, notch_fs = 50, notch_q = 30):
    filt_out = np.zeros_like(data)
    for i in range(data.shape[0]):
        filt_out[i,:] = notch_filter(butter_bandpass_filter(data[i,:], lowcut, highcut, fs, order=order), notch_fs, fs, notch_q)
    return filt_out


# In[2]:


with h5py.File('increasingforce1.mat', 'r') as file:
    # List all groups
    print("Keys: %s" % file.keys())
    # Get the data
    increase_grid_one = file['grid_crds'][()].T
    increase_mat_one = file['out_mat'][()].T
    
with h5py.File('increasingforce2.mat', 'r') as file:
    # List all groups
    print("Keys: %s" % file.keys())
    # Get the data
    increase_grid_two = file['grid_crds'][()].T
    increase_mat_two = file['out_mat'][()].T
    
with h5py.File('steadyforce.mat', 'r') as file:
    # List all groups
    print("Keys: %s" % file.keys())
    # Get the data
    steady_grid = file['grid_crds'][()].T
    steady_mat = file['out_mat'][()].T
    
    
filted_increase_mat_one = filt_GRID(increase_mat_one)
filted_increase_mat_two = filt_GRID(increase_mat_two)
filted_steady_mat = filt_GRID(steady_mat)


print(filted_steady_mat.shape)
print(filted_increase_mat_one.shape)
print(filted_increase_mat_two.shape)


np.savetxt(
    'increase_mat_one.csv',
    filted_increase_mat_one, 
    delimiter=','
)

np.savetxt(
    'increase_mat_two.csv',
    filted_increase_mat_two,
    delimiter=','
)

np.savetxt(
    'steady_mat.csv',
    filted_steady_mat,
    delimiter=','
)


# In[3]:


import matplotlib.pyplot as plt

plt.figure(figsize=(20, 150))

for each_basis in range(filted_steady_mat.shape[0]):
    ax = plt.subplot(filted_steady_mat.shape[0], 1, each_basis + 1)
    plt.plot(filted_steady_mat[each_basis])
    ax.set_title(f"Component NO. {each_basis + 1}")
    
plt.subplots_adjust(wspace=0.1, hspace=0.4) 
plt.show()


# ### Peak Detection

# In[13]:


from tqdm import tqdm 

def AMPD(data):
    """
    param data: 1-D numpy.ndarray 
    return:     peak indices
    
    Reference:  https://zhuanlan.zhihu.com/p/549588865
    """
    p_data = np.zeros_like(data, dtype=np.int32)
    count = data.shape[0]
    arr_rowsum = []
    
    for k in tqdm(range(1, count // 2 + 1)):
        row_sum = 0
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                row_sum -= 1
                
        arr_rowsum.append(row_sum)
        
    min_index = np.argmin(arr_rowsum)
    max_window_length = min_index
    
    for k in tqdm(range(1, max_window_length + 1)):
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                p_data[i] += 1
                
    return np.where(p_data == max_window_length)[0]


# In[14]:


## TOO SLOW

# all_peak_indices = []
# for each_base in range(filted_steady_mat.shape[0]):
#     peak_indices = AMPD(filted_steady_mat[each_base])
#     print(len(peak_indices))
#     all_peak_indices.append(np.array(peak_indices).reshape(-1, 1))


# In[15]:


from scipy.signal import find_peaks


plt.figure(figsize=(20, 150))

peaks_indices = []
for each_basis in range(filted_steady_mat.shape[0]):
    peaks, _ = find_peaks(filted_steady_mat[each_basis], height=0, distance=100, threshold=None)
    peaks = peaks.reshape(-1, 1)
    peaks_indices.append(peaks)
    if each_basis == 0:
        print("For the first basis, peaks indices.shape: ", peaks.shape)
    if each_basis == 1:
        print("For the second basis, peaks indices.shape: ", peaks.shape)
    
    ax = plt.subplot(filted_steady_mat.shape[0], 1, each_basis + 1)
    plt.plot(filted_steady_mat[each_basis])
    plt.scatter(peaks, filted_steady_mat[each_basis][peaks], marker='x', color='r')
    ax.set_title(f"Component NO. {each_basis + 1}")
    
plt.subplots_adjust(wspace=0.1, hspace=0.4) 
plt.show()


# ##### For the first basis, it has `peaks_indices[0]`, slice its peak series.

# In[16]:


A = filted_steady_mat[0].reshape(-1, 1)
print(A.shape)
B = peaks_indices[0]
print(B.shape)

window_size = 50
windows = []

for index in B.flatten():
    # B.flatten() is a list of indices
    start_index = max(0, index - window_size // 2)
    end_index = min(A.shape[0], index + window_size // 2)

    window = A[start_index:end_index] # (50, 1)
    windows.append(window)

window_matrix_tmp = np.array(windows) # (530, 50, 1)
# print(window_matrix_tmp)
window_matrix = np.squeeze(window_matrix_tmp, axis=-1) # (530, 50)
print(window_matrix.shape)
# print(window_matrix)


# ##### PCA to reduce the dimension and K-Means to cluster (gain the kinds of the Motor Units / peak windows)

# In[17]:


# PCA to reduce the dimension of the window matrix
from sklearn.decomposition import PCA

pca_model = PCA(n_components=2)
window_matrix_pca = pca_model.fit_transform(window_matrix) # (530, 2)


# K-means clustering (better for lower dimension data)
from sklearn.cluster import KMeans


# Without a priori knowledge of the number of clusters, so evaluate the clustering performance first
max_clusters = 40
inertia = []

for n_clusters in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(window_matrix_pca)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, max_clusters + 1), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

# Choose K = 10
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(window_matrix_pca)

plt.scatter(window_matrix_pca[:, 0], window_matrix_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.title('K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# In[18]:


mu_indices_list = []
for i in range(num_clusters):
    mu_indices = []
    for j in range(window_matrix_pca.shape[0]):
        if cluster_labels[j] == i:
            mu_indices.append(j)
            
    mu_indices_list.append(mu_indices)

print(len(mu_indices_list[1]))


# In[19]:


plt.figure(figsize=(20, 150))

for each_basis in range(len(mu_indices_list[1])):
    ax = plt.subplot(len(mu_indices_list[1]), 1, each_basis + 1)
    plt.plot(window_matrix[mu_indices_list[1][each_basis]])
    ax.set_title(f"Label Peak NO. {each_basis + 1}")
    
plt.subplots_adjust(wspace=0.1, hspace=0.4) 
plt.show()


# ##### For all channels in steady, do the same as the toy demo did before:

# In[20]:


window_size = 50
windows_64ch = []

for each_ch in range(filted_steady_mat.shape[0]):
    
    windows = []
    A = filted_steady_mat[each_ch].reshape(-1, 1)
    B = peaks_indices[each_ch]

    for index in B.flatten():

        start_index = max(0, index - window_size // 2)
        end_index = min(A.shape[0], index + window_size // 2)
        if end_index - start_index < window_size:
            continue
            
        window = A[start_index:end_index] # (50, 1)
        windows.append(window)
        
    window_matrix_tmp = np.array(windows) # (530, 50, 1)
    window_matrix = np.squeeze(window_matrix_tmp, axis=-1) # (530, 50)
    windows_64ch.append(window_matrix)


# In[21]:


win_reduced_64ch = []
cluster_repre_64ch = []
cluster_labels_64ch = []

for each_ch_win in range(len(windows_64ch)):
    # PCA dimension reduction
    pca_model = PCA(n_components=2)
    window_matrix_pca = pca_model.fit_transform(windows_64ch[each_ch_win]) # (530, 2)
    win_reduced_64ch.append(window_matrix_pca)
    
    # K-means clustering
    num_clusters = 10
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(window_matrix_pca) # (530, )
    
    cluster_label_indices = []
    cluster_repre = []
    # To compute a mean / representative waveform
    for i in range(num_clusters):
        each_group_wave = []
        label_indices = []
        for j in range(window_matrix_pca.shape[0]):
            if cluster_labels[j] == i:
                each_group_wave.append(windows_64ch[each_ch_win][j])
                label_indices.append(j)
                
        # print(len(each_group_wave))
        each_group_wave_array = np.array(each_group_wave) # (num_per_group, 50)
        representative_win = np.mean(each_group_wave_array, axis=0) # (50, )
        cluster_repre.append(representative_win)
        cluster_label_indices.append(label_indices)
    
    cluster_repre_array = np.array(cluster_repre) # (10, 50)
    cluster_repre_64ch.append(cluster_repre_array)
    cluster_labels_64ch.append(cluster_label_indices)


cluster_repre_64ch_array = np.array(cluster_repre_64ch) # (64, 10, 50)
# For cluster_labels_64ch,
# cluster_labels_64ch[0] is a list of all labels' indices of the first channel
# cluster_labels_64ch[0][0] is a list of indices of the first label of the first channel


# In[22]:


get_ipython().run_line_magic('pip', 'install fastdtw')
get_ipython().run_line_magic('pip', 'install dtw-python # plot the dtw path')

from dtw import *
import fastdtw
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import cosine_similarity

def calculate_dtw_distance(series1, series2):
    distance = fastdtw.fastdtw(series1, series2)
    return distance[0]


# In[23]:


matrix_shape = (
    cluster_repre_64ch_array.shape[0],
    cluster_repre_64ch_array.shape[1],
    cluster_repre_64ch_array.shape[0],
    cluster_repre_64ch_array.shape[1]
)
dtw_similarity = np.zeros(matrix_shape)
cos_similarity = np.zeros(matrix_shape)
cov_similarity = np.zeros(matrix_shape)


for ch in range(cluster_repre_64ch_array.shape[0]):
    for label in range(cluster_repre_64ch_array.shape[1]):
        
        # The baseline is cluster_repre_64ch_array[ch, label]
        for other_ch in range(cluster_repre_64ch_array.shape[0]):
            if other_ch <= ch:
                continue

            for label_other_ch in range(cluster_repre_64ch_array.shape[1]):
                # print(cluster_repre_64ch_array[ch, label].shape)
                # print(f"Channel {ch} Label {label} is compared with Channel {other_ch} Label {label_other_ch}")
                
                similarity = calculate_dtw_distance(
                    cluster_repre_64ch_array[ch, label], 
                    cluster_repre_64ch_array[other_ch, label_other_ch]
                )
                dtw_similarity[ch, label, other_ch, label_other_ch] = similarity
                
                # Just plot one for DTW distance
                if label == 0 and label_other_ch == 0 and ch == 0 and other_ch == 1:
                    #Reference: @https://dynamictimewarping.github.io/python/#the-alignment-class
                    alignment = dtw(cluster_repre_64ch_array[ch, label], cluster_repre_64ch_array[other_ch, label_other_ch], keep_internals=True)
                    dtw(cluster_repre_64ch_array[ch, label],
                        cluster_repre_64ch_array[other_ch, label_other_ch],
                        keep_internals=True, 
                        step_pattern=rabinerJuangStepPattern(6, "c")
                    ).plot(type="twoway",offset=-2)
                
                # Other metrics for the similarity
                cosine = cosine_similarity(
                    cluster_repre_64ch_array[ch, label].reshape(1, -1), 
                    cluster_repre_64ch_array[other_ch, label_other_ch].reshape(1, -1)
                )
                cos_similarity[ch, label, other_ch, label_other_ch] = cosine[0][0]
                
                # Cov would 2 by 2, diagonal elements are 1, non-diagonal elements are the same
                cov = np.corrcoef(
                    cluster_repre_64ch_array[ch, label], 
                    cluster_repre_64ch_array[other_ch, label_other_ch]
                )
                cov_similarity[ch, label, other_ch, label_other_ch] = cov[0][1]
                
                print(f"[{ch}:{label}]<>[{other_ch}:{label_other_ch}]", similarity, cosine[0][0], cov[0][1])
                
                # not cross-correlation
                # not correlation
                
                # Final evaluation metric statemnet
                
                # Intersection hyperparameter
                # distance of the peaks hyperparameter
                # threshold of the similarities hyperparameter
                # range of valid firing instance moments
                


# In[24]:


# save as bin 
np.save('dtw_matrix_4D.npy', dtw_similarity)
np.save('cos_matrix_4D.npy', cos_similarity)
np.save('cov_matrix_4D.npy', cov_similarity)


# In[25]:


import numpy as np
loaded_dtw = np.load('dtw_matrix_4D.npy')
loaded_cos = np.load('cos_matrix_4D.npy')
loaded_cov = np.load('cov_matrix_4D.npy')


# In[26]:


loaded_cos.shape


# ##### Pick up the matching MUs

# In[27]:


list_similar_labels = []
threshold_for_cov = 0.98

for ch in range(loaded_cos.shape[0]):
    
    if ch == loaded_cos.shape[0] - 1:
        break
    
    
    list_similar_for_each_label = []
    for label in range(loaded_cos.shape[1]):
        
        # For each lable in baseline channel, find the most similar label in other channels (#num of channels (64 - ch - 1))
        
        matching_MU_label_from_each_ch = np.zeros((loaded_cos.shape[0] - ch - 1, 1))
        for other_ch in range(loaded_cos.shape[2]):
            if other_ch <= ch:
                continue
            
            # Threshold of the similarities would be hyperparameter
            # Merge two labels in other channels if cos is higher than a threshold
            highest_cov = -1
            for label_other_ch in range(loaded_cos.shape[3]):
                # print(loaded_cov[ch, label, other_ch, label_other_ch])
                
                if loaded_cov[ch, label, other_ch, label_other_ch] > highest_cov:
                    highest_cov = loaded_cov[ch, label, other_ch, label_other_ch]
                    if highest_cov <= threshold_for_cov:
                        matching_MU_label_from_each_ch[other_ch - ch - 1] = -1
                        continue
                    
                    if other_ch == 1 and ch == 0 and label == 0:
                        # print("When label == 0 in ch == 0 and other labels in channel 1 ", highest_cov) 
                        # print(highest_cov_label)
                        pass
                    
                    highest_cov_label = label_other_ch
                    matching_MU_label_from_each_ch[other_ch - ch - 1] = highest_cov_label
        
        
        
        # print(matching_MU_label_from_each_ch.shape)
        # print(matching_MU_label_from_each_ch)            
        list_similar_for_each_label.append(matching_MU_label_from_each_ch)
    
    
    # print(len(list_similar_for_each_label)) # 10
    list_similar_labels.append(list_similar_for_each_label)

# print(len(list_similar_labels)) # 63
                    


# In[28]:


len(list_similar_labels[1][1])


# In[29]:


list_similar_labels[0][0]


# ##### Finding the firing moments for MU one in channel one

# In[30]:


# print(cluster_repre_64ch_array) # representative waveforms
# print(cluster_labels_64ch) # indices list of each label in each channel (64, 10)
# print(cluster_labels_64ch[0][0])

channel_first_label_first = cluster_labels_64ch[0][0]
MU_label_indices_in_other_channels_for_first_channel_first_label = list_similar_labels[0][0] # (63, 1)

# Get the MU label indices, for each other channel(except the first channel)
list_MU_indices_in_other_channels_for_first_channel_first_label = []
for i in range(MU_label_indices_in_other_channels_for_first_channel_first_label.shape[0]): # 63
    if MU_label_indices_in_other_channels_for_first_channel_first_label[i] == -1:
        list_MU_indices_in_other_channels_for_first_channel_first_label.append(-1)
        continue
    
    # print(i + 1)
    # print(MU_label_indices_in_other_channels_for_first_channel_first_label[i]) # (1, )
    
    list_MU_indices_in_other_channels_for_first_channel_first_label.append(
        cluster_labels_64ch[i + 1][(int)(MU_label_indices_in_other_channels_for_first_channel_first_label[i][0])]
    )

print(list_MU_indices_in_other_channels_for_first_channel_first_label)
print(len(list_MU_indices_in_other_channels_for_first_channel_first_label)) # (63)


# In[31]:


len(list_MU_indices_in_other_channels_for_first_channel_first_label[0])


# ##### Put each matching MU into plot and find out the firing moments

# In[32]:


def peaks2timespan(peak_moments, filted_steady_mat_width):
    """
    Given peak moments indices (vector), and filtered steady mat width (time series length e.g. 700000) as the maximum boundry
    Return a list of tuples, each tuple is a time span pair (start_index, end_index)
    Time span is window size of 50, centered at the peak moment, as the form of a pair above.
    (Ignore the overlap between two time spans, which just increases the opacity of the time span in the plot if any (overlap).)
    """
    window_size = 50
    list_indices = []
    
    for i in range(len(peak_moments)):
        start_index = max(0, peak_moments[i] - window_size // 2)
        end_index = min(filted_steady_mat_width, peak_moments[i] + window_size // 2)
        
        # print(list(range(start_index, end_index)))
        # list_indices += list(range(start_index, end_index))
        
        list_indices.append((start_index, end_index))
        
    return list_indices


# In[33]:


# peaks_moments = []
# print(len(peaks_indices[1]), np.squeeze(peaks_indices[1]).shape)
# print()
# for i in range(len(list_MU_indices_in_other_channels_for_first_channel_first_label[0])):
#     # print(list_MU_indices_in_other_channels_for_first_channel_first_label[0][i]) # the indices of peaks in peak indices
#     peaks_moments.append(peaks_indices[1][list_MU_indices_in_other_channels_for_first_channel_first_label[0][i]][0])

# print(peaks_moments)

second_channel_peaks = np.squeeze(peaks_indices[1][list_MU_indices_in_other_channels_for_first_channel_first_label[0]])
final_span_second_channel = peaks2timespan(second_channel_peaks, filted_steady_mat.shape[1])
# len(final_span_second_channel)
print(len(peaks_indices[0]))
print(np.squeeze(peaks_indices[0]).tolist())
final_span_first_channel = peaks2timespan(np.squeeze(peaks_indices[0][cluster_labels_64ch[0][0]]).tolist(), filted_steady_mat.shape[1])
len(final_span_first_channel)


# In[34]:


plt.figure(figsize=(20, 5))

for each_basis in range(2):
    ax = plt.subplot(2, 1, each_basis + 1)
    plt.plot(filted_steady_mat[each_basis])
    
    if each_basis == 0:
        cnt = 0
        for start, end in final_span_first_channel:
            cnt = cnt + 1
            plt.axvspan(start, end, color='red', alpha=0.3)
        print(cnt)
        
    elif each_basis == 1:   
        cntt = 0
        for start, end in final_span_second_channel:
            cntt = cntt + 1
            plt.axvspan(start, end, color='red', alpha=0.3)
        print(cntt)
        
    ax.set_title(f"Channel(Basis) NO. {each_basis + 1}")
    
plt.subplots_adjust(wspace=0.1, hspace=0.4) 
plt.show()


# ##### Compare the first MU(peak waveform) in the first channel with this kind of MUs in all other 63 channels.

# In[35]:


final_channels_timespan = []
final_span_first_channel = peaks2timespan(
    np.squeeze(peaks_indices[0][cluster_labels_64ch[0][0]]).tolist(), 
    filted_steady_mat.shape[1]
)
final_channels_timespan.append(final_span_first_channel)


for other_channels in range(len(list_MU_indices_in_other_channels_for_first_channel_first_label)):
    if list_MU_indices_in_other_channels_for_first_channel_first_label[other_channels] == -1:
        final_channels_timespan.append(-1)
        continue
    
    peaks_this_channel = np.squeeze(
        peaks_indices[other_channels + 1][list_MU_indices_in_other_channels_for_first_channel_first_label[other_channels]]
    )
    final_span_this_channel = peaks2timespan(
        peaks_this_channel, 
        filted_steady_mat.shape[1]
    )
    final_channels_timespan.append(final_span_this_channel)
    
    # print(len(final_span_this_channel)) # 16 length numbers because only 16 labels found to match this first MU in the first channel
    
# print(len(final_channels_timespan))
# print(final_channels_timespan[63])



# Plot
plt.figure(figsize=(20, 160))

for each_basis in range(filted_steady_mat.shape[0]):
    ax = plt.subplot(filted_steady_mat.shape[0], 1, each_basis + 1)
    plt.plot(filted_steady_mat[each_basis])
    
    if final_channels_timespan[each_basis] == -1:
        # No matching MU (peak waveform) found in this channel
        ax.set_title(f"Channel(Basis) NO. {each_basis + 1}")
        continue
    
    for start, end in final_channels_timespan[each_basis]:
        plt.axvspan(start, end, color='red', alpha=0.3)
        
    ax.set_title(f"Channel(Basis) NO. {each_basis + 1}")
    
plt.subplots_adjust(wspace=0.1, hspace=0.4) 
plt.show()


# ### For MUs in the first channel, find their matching MUs in other channels respectively.

# In[36]:


print(np.squeeze(list_similar_labels[0][0]))
print(np.squeeze(list_similar_labels[0][1]))
print(np.squeeze(list_similar_labels[0][2]))
print(np.squeeze(list_similar_labels[0][9]))


# In[107]:


first_channel_all_labels = []
MU_labels_in_other_channels_for_first_ch_all_labels = []

for each_label in range(len(cluster_labels_64ch[0])):
    first_channel_all_labels.append(
        cluster_labels_64ch[0][each_label]
    ) 
    
    MU_labels_in_other_channels_for_first_ch_all_labels.append(
        np.squeeze(list_similar_labels[0][each_label])
    ) # 10 x (63, 1)  
    
    # Get the MU label indices, for each other channel(except the first channel)
    list_MU_indices_in_other_channels_for_first_channel_all_labels = []
    for i in range(len(MU_labels_in_other_channels_for_first_ch_all_labels)): # 10

        list_MU_indices_in_other_channels_for_first_channel_each_label = []

        for j in range(MU_labels_in_other_channels_for_first_ch_all_labels[i].shape[0]): # 63
            if MU_labels_in_other_channels_for_first_ch_all_labels[i][j] == -1:
                list_MU_indices_in_other_channels_for_first_channel_each_label.append(-1)
                continue

            # j + # NO. of current channel
            list_MU_indices_in_other_channels_for_first_channel_each_label.append(
                cluster_labels_64ch[j + 1 ][(int)(MU_labels_in_other_channels_for_first_ch_all_labels[i][j])]
            ) 

        # print(list_MU_indices_in_other_channels_for_first_channel_each_label)
        print(len(list_MU_indices_in_other_channels_for_first_channel_each_label)) # 63
        list_MU_indices_in_other_channels_for_first_channel_all_labels.append(
            list_MU_indices_in_other_channels_for_first_channel_each_label
        )

print(len(list_MU_indices_in_other_channels_for_first_channel_all_labels)) # 10 


# In[74]:


all_channels_labels = []
all_channels_MU_labels_in_other_channels = []
list_MU_indices_in_other_channels_for_current_channel_all_labels = []
list_MU_indices_in_other_channels_for_current_channel_each_label = []

for current_channel in range(62):
    current_channel_all_labels = []
    MU_labels_in_other_channels_for_current_ch_all_labels = []

    for each_label in range(len(cluster_labels_64ch[current_channel])):
        current_channel_all_labels.append(cluster_labels_64ch[current_channel][each_label])
        
        MU_labels_in_other_channels_for_current_ch_all_labels.append(np.squeeze(list_similar_labels[current_channel][each_label]))

        
        for i in range(len(MU_labels_in_other_channels_for_current_ch_all_labels)):
            

            for j in range(MU_labels_in_other_channels_for_current_ch_all_labels[i].shape[0]):
                if MU_labels_in_other_channels_for_current_ch_all_labels[i][j] == -1:
                    list_MU_indices_in_other_channels_for_current_channel_each_label.append(-1)
                    continue

                next_channel = (j + current_channel + 1) % 64  # To loop back to the first channel after the 64th
                list_MU_indices_in_other_channels_for_current_channel_each_label.append(
                    cluster_labels_64ch[next_channel][(int)(MU_labels_in_other_channels_for_current_ch_all_labels[i][j])]
                )

            list_MU_indices_in_other_channels_for_current_channel_all_labels.append(list_MU_indices_in_other_channels_for_current_channel_each_label)

    all_channels_labels.append(current_channel_all_labels)
    all_channels_MU_labels_in_other_channels.append(list_MU_indices_in_other_channels_for_current_channel_all_labels)


# In[42]:


len(peaks_indices[1])


# In[108]:


all_channels_timespan_all_labels = []

for i in range(0,63):
    for each_lb in range(len(list_MU_indices_in_other_channels_for_first_channel_all_labels)):
        final_channels_timespan_each_label = []
        final_channels_timespan_each_label.append(final_span_first_channel)

        for other_channels in range(len(list_MU_indices_in_other_channels_for_first_channel_all_labels[each_lb])):
            if list_MU_indices_in_other_channels_for_first_channel_all_labels[each_lb][other_channels] == -1:
                final_channels_timespan_each_label.append(-1)
                continue

            peaks_this_channel = np.squeeze(
                peaks_indices[other_channels + 1][list_MU_indices_in_other_channels_for_first_channel_all_labels[each_lb][other_channels]]
            )

            # Check if peaks_this_channel is a scalar or empty, and handle accordingly
            if np.isscalar(peaks_this_channel) or peaks_this_channel.size == 0:
                final_channels_timespan_each_label.append(-1)
                continue

            if peaks_this_channel.ndim == 0:  # Handling for a single value
                peaks_this_channel = [peaks_this_channel.item()]

            final_span_this_channel = peaks2timespan(
                peaks_this_channel, 
                filted_steady_mat.shape[1]
            )
            final_channels_timespan_each_label.append(final_span_this_channel)

        final_channels_timespan_all_labels.append(final_channels_timespan_each_label)


# In[90]:


total_display_basis = 0
useful_case = []
for the_label in range(len(final_channels_timespan_all_labels)):
    useful = -1
    for i in range(len(final_channels_timespan_all_labels[the_label])):
        if final_channels_timespan_all_labels[the_label][i] != -1:
            useful = useful + 1
            
    if useful != 0:
        total_display_basis = total_display_basis + useful + 1     
    useful_case.append(useful)
    
print(useful_case)
print(total_display_basis)


# In[104]:


print(len(final_channels_timespan_all_labels[0][0][0]))


# In[ ]:


diff_label_colors_for_ch1 = ['blue', 'green', 'cyan', 'yellow', 'lightgray', 'pink', 'brown', 'orange']
# Plot
plt.figure(figsize=(20, total_display_basis * 2))

display_cnt = 0
color_cnt = -1
# for Channel One
for each_label_case in range(len(final_channels_timespan_all_labels)): # 10
    
    if useful_case[each_label_case] == 0:
        continue
    color_cnt = color_cnt + 1 # just for rendering the color for useful channels 
    
    
    for each_matching_basis in range(len(final_channels_timespan_all_labels[0])): # 64
    
        if final_channels_timespan_all_labels[each_label_case][each_matching_basis] == -1:
            # No matching MU (peak waveform) found in this channel
            continue
        
        # print(display_cnt)
        ax = plt.subplot(total_display_basis, 1, display_cnt + 1)
        display_cnt = display_cnt + 1

        plt.plot(
            filted_steady_mat[each_matching_basis], 
            color=diff_label_colors_for_ch1[color_cnt % len(diff_label_colors_for_ch1)]
        )
        
        for start, end in final_channels_timespan_all_labels[each_label_case][each_matching_basis]:
            plt.axvspan(start, end, color='red', alpha=0.3)
            
        ax.set_title(f"For Label {each_label_case + 1} in CH1, the matching MU in Channel(Basis) NO. {each_matching_basis + 1}")
    
plt.subplots_adjust(wspace=0.1, hspace=0.8) 
plt.show()


# In[ ]:




