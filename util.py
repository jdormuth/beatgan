import numpy as np
import pretty_midi
import librosa
import os
import fnmatch
import midi2pianoroll

def write_piano_roll_to_midi(piano_roll, filename, program_num=0, is_drum=False, velocity=100, tempo=120.0, beat_resolution=24):

    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Create an Instrument object
    instrument = pretty_midi.Instrument(program=program_num, is_drum=is_drum)
    # Set the piano roll to the Instrument object
    set_piano_roll_to_instrument(piano_roll, instrument, velocity, tempo, beat_resolution)
    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument)
    # Write out the MIDI data
    midi.write(filename)

def set_piano_roll_to_instrument(piano_roll, instrument, velocity=100, tempo=120.0, beat_resolution=24):
    # Calculate time per pixel
    tpp = 60.0/tempo/float(beat_resolution)
    
    # Create piano_roll_search that captures note onsets and offsets
    piano_roll = piano_roll.reshape((piano_roll.shape[0] * piano_roll.shape[1], piano_roll.shape[2]))
    
    piano_roll_diff = np.concatenate((np.zeros((1,10),dtype=int), piano_roll, np.zeros((1,10),dtype=int)))  
    piano_roll_search = np.diff(piano_roll_diff.astype(int), axis=0)

    # Iterate through all possible(128) pitches
    for note_num in range(10):
        # Search for notes
        start_idx = (piano_roll_search[:,note_num] > 0).nonzero()
        start_time = tpp*(start_idx[0].astype(float))
        end_idx = (piano_roll_search[:,note_num] < 0).nonzero()
        end_time = tpp*(end_idx[0].astype(float))
        # Iterate through all the searched notes
        for idx in range(len(start_time)):
            # Create an Note object with corresponding note number, start time and end time
            note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx], end=end_time[idx])
            # Add the note to the Instrument object
            instrument.notes.append(note)
    # Sort the notes by their start time
    instrument.notes.sort(key=lambda note: note.start)

#helper function for future for help with loading in midi files and converting to numpy arrays
def load_midi(filepath):
    midi_dict = {}
    for root, dirnames, filenames in os.walk(filepath):
        for filename in fnmatch.filter(filenames, '*.mid'):
            try:
                midi_dict[root].append(root+'/'+filename)
            except:
                midi_dict[root] = root+'/'+filename
    return midi_dict

def get_midi(filepath):
    numpy_directory = './numpy_samples'
    if not os.path.exists(numpy_directory):
        os.makedirs(numpy_directory)
    
    for root, dirs, files in os.walk(filepath):
        x = np.array([np.array(midi2pianoroll.midi_to_pianorolls(root+"/"+file)[0][0]) for file in files])
        maximum = 0
        minimum = 1000
        for i in range(x.shape[0]):
            
            x[i] = x[i][:96]
            for j in range(x[i].shape[0]):
                for k in range(x[i][j].shape[0]):
                    if(x[i][j][k] > 0):
                        if(k > maximum):
                            maximum = k
                        if(k < minimum):
                            minimum = k
        x_new = np.concatenate(x, axis=0)
        
        x_final = np.reshape(x_new,(-1,96,128))
        np.save(numpy_directory+"/"+"x_beats",x_final)
        
#area to run test scripts for above functions
get_midi('./jacob_midis')