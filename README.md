# Mouse Initialization

We pretrain a neural network to replicate the brain activity of a mouse for improved performance. The brain activity data is extracted from https://github.com/int-brain-lab/paper-brain-wide-map. The baseline for the DQN is from https://raw.githubusercontent.com/seungeunrho/minimalRL/refs/heads/master/dqn.py.

## Total Rewards Per 20 Episodes (Averaged Across 100 Runs) (Max 1000 Steps Per Episode) (500 Episodes Per Run)

<img width="640" height="480" alt="mouse_plot" src="https://github.com/user-attachments/assets/677c1b63-f4cd-4bd6-ba33-cd351224b404" />

## Data Extraction Details

We use the Collab Notebook provided at https://github.com/int-brain-lab/paper-brain-wide-map. Then we add a cell around 1/4 of the through the notebook with the following code:

```py
for pid in df_bw['pid']:
  [eid, pname] = one.pid2eid(pid)
  ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
  spikes, clusters, channels = ssl.load_spike_sorting()
  # print(spikes.shape, clusters.shape, channels.shape)
  clusters = ssl.merge_clusters(spikes, clusters, channels)
  print(clusters[['channels', 'amp_median']])
  break # only use one pid
```
then,
```py
channels = clusters['channels']
spikes = clusters['amp_median']

# Initialize an empty dictionary to store the results
spikes_by_channel = {}

# Iterate through each pair of channel and spike
for channel, spike in zip(channels, spikes):
    # If the channel isn't already a key in the dictionary, add it with an empty list
    if channel not in spikes_by_channel:
        spikes_by_channel[channel] = spike # only use first spike
    # Append the current spike to the list for its channel
        # spikes_by_channel[channel] = spike
```
That is, while the data is inherently temporal, because of its irregularity we treat it as one time step.
