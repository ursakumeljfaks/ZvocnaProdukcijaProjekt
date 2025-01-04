import struct
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import librosa
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import wave
import pygame

################################# 2nd APPROACH

def audio_to_midi_orig(audio_path, midi_path, min_note_duration=0.2):
    """
    Convert audio to MIDI dynamically for any input audio.
    
    Parameters:
    - audio_path: Path to the input audio file.
    - midi_path: Path to save the output MIDI file.
    - min_note_duration: Minimum duration of the notes to prevent small MIDI notes.

    Returns:
    - MIDI file
    """
    # Load the audio file
    # y = amplitudes
    # sr = sampling rate of the audio in Hz
    y, sr = librosa.load(audio_path, sr=None)
    
    # YIN algorithm to extract fundamental frequencies
    # f0 = 1D array of fundamental frequencies over time
    # A0 ~ 27.5Hz lowest piano note
    # C8 ~ 4186Hz highest piano note
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('A0'), fmax=librosa.note_to_hz('C8'))
    # Convert f0 frequencies to a piano roll
    piano_roll = np.zeros((len(f0), 128))  # 128 MIDI notes
    for t, pitch_hz in enumerate(f0):
        # If the note is not silent, if it is f0 gives you NaN value
        if not np.isnan(pitch_hz):
            note_number = int(np.round(librosa.hz_to_midi(pitch_hz)))
            if 0 <= note_number < 128: 
                # Note is active at time frame t and at note nb note_number
                piano_roll[t, note_number] = 1

    # Convert the piano roll to MIDI notes
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0) # 0 is piano
    time_per_frame = librosa.frames_to_time(1, sr=sr) # Converts the frame duration of 1 frame into seconds
    
    for note_number in range(128):
        on = False
        start_time = 0
        for t in range(piano_roll.shape[0]):
            # Cote has not been played yet but is active
            if piano_roll[t, note_number] == 1 and not on:
                on = True
                start_time = t * time_per_frame
            # Cote is no longer active
            elif piano_roll[t, note_number] == 0 and on:
                on = False
                end_time = t * time_per_frame
                # Check threshold to avoid non musical notes
                if end_time - start_time >= min_note_duration:
                    note = pretty_midi.Note(velocity=120, pitch=note_number, start=start_time, end=end_time)
                    instrument.notes.append(note)
        if on:  # Handle last note
            end_time = piano_roll.shape[0] * time_per_frame
            if end_time - start_time >= min_note_duration:
                note = pretty_midi.Note(velocity=120, pitch=note_number, start=start_time, end=end_time)
                instrument.notes.append(note)
    
    # Adding the instrument to the midi objects and saving it
    midi.instruments.append(instrument)
    for instrument in midi.instruments:
        for note in instrument.notes:
            print(note.pitch)

    midi.write(midi_path)

##################### 1st APPROACH = DETECTING ONE NOTEE FROM A AUDIO FILE
def note_detect(audio_file):
    """
    Convert audio to closest detected note and then this note converting to MIDI note. It works only for one note in an audio file.
    
    Parameters:
    - audio_file: Path to the input audio file.

    Returns:
    - list [closest_detected_note, its_frequency]
    """

    name = np.array(["C0","C#0","D0","D#0","E0","F0","F#0","G0","G#0","A0","A#0","B0","C1","C#1","D1","D#1","E1","F1","F#1","G1","G#1","A1","A#1","B1","C2","C#2","D2","D#2","E2","F2","F#2","G2","G2#","A2","A2#","B2","C3","C3#","D3","D3#","E3","F3","F3#","G3","G3#","A3","A3#","B3","C4","C4#","D4","D4#","E4","F4","F4#","G4","G4#","A4","A4#","B4","C5","C5#","D5","D5#","E5","F5","F5#","G5","G#5","A5","A5#","B5","C6","C6#","D6","D6#","E6","F6","F6#","G6","G6#","A6","A6#","B6","C7","C7#","D7","D7#","E7","F7","F7#","G7","G7#","A7","A7#","B7","C8","C8#","D8","D8#","E8","F8","F8#","G8","G8#","A8","A8#","B8","Beyond B8"])
    frequencies = np.array([16.35,17.32,18.35,19.45,20.60,21.83,23.12,24.50,25.96,27.50,29.14,30.87,32.70,34.65,36.71,38.89,41.20,43.65,46.25,49.00,51.91,55.00,58.27,61.74,65.41,69.30,73.42,77.78,82.41,87.31,92.50,98.00,103.83,110.00,116.54,123.47,130.81,138.59,146.83,155.56,164.81,174.61,185.00,196.00,207.65,220.00,233.08,246.94,261.63,277.18,293.66,311.13,329.63,349.23,369.99,392.00,415.30,440.00,466.16,493.88,523.25,554.37,587.33,622.25,659.26,698.46,739.99,783.99,830.61,880.00,932.33,987.77,1046.50,1108.73,1174.66,1244.51,1318.51,1396.91,1479.98,1567.98,1661.22,1760.00,1864.66,1975.53,2093.00,2217.46,2349.32,2489.02,2637.02,2793.83,2959.96,3135.96,3322.44,3520.00,3729.31,3951.07,4186.01,4434.92,4698.64,4978.03,5274.04,5587.65,5919.91,6271.93,6644.88,7040.00,7458.62,7902.13,8000])

    # Convert to dictionary
    note_frequencies = dict(zip(name, frequencies))
    # Open the audio file using wave module
    with wave.open(audio_file, 'rb') as audio:
        # Extract audio properties
        file_length = audio.getnframes() 
        f_s = audio.getframerate()  # Sampling frequency
        n_channels = audio.getnchannels()  # Number of channels (mono = 1, stereo = 2)
        sample_width = audio.getsampwidth()  # Sample width in bytes (1 byte for 8-bit, 2 bytes for 16-bit)
        
        # Calculate the frame size (sample width * number of channels)
        frame_size = sample_width * n_channels
        
        # Create an empty array to store sound data
        sound = np.zeros(file_length) 
        
        # Read audio data from the file
        for i in range(file_length):
            wdata = audio.readframes(1)
            
            if len(wdata) == frame_size:  # Ensure the correct number of bytes are read per frame
                if sample_width == 2:  # 16-bit audio (2 bytes per sample)
                    data = struct.unpack("<h", wdata[0:2])  # Unpack as 16-bit integer
                    sound[i] = data[0]
                elif sample_width == 1:  # 8-bit audio (1 byte per sample)
                    data = struct.unpack("<B", wdata[0:1])  # Unpack as 8-bit unsigned integer
                    sound[i] = data[0]
            else:
                print(f"Warning: Skipping a frame because of incorrect data length: {len(wdata)}")
        
        # Normalize sound data (important for 16-bit audio)
        sound = np.divide(sound, float(2**15))  # Scale to -1 to 1 for 16-bit audio

        # Fourier transform to detect frequencies
        fourier = np.fft.fft(sound)
        fourier = np.absolute(fourier)  # Get magnitude of complex numbers
        
        # Plot the Fourier transform (frequency spectrum)
        # plt.plot(fourier)
        # plt.title("Frequency Spectrum")
        # plt.xlabel("Frequency Bin")
        # plt.ylabel("Magnitude")
        # plt.show()

        # Find the index of the most powerful frequency (highest magnitude)
        peak_index = np.argmax(fourier[:file_length // 2])  # Only consider the positive frequencies
        
        # Convert the index to the actual frequency
        peak_freq = (peak_index * f_s) / file_length  # Frequency formula: (index * sampling_rate) / length_of_signal

        # Find the closest note
        closest_note = min(note_frequencies, key=lambda note: abs(note_frequencies[note] - peak_freq))
        
        # Return the most powerful frequency and the corresponding note
        print("Detected Frequency:", peak_freq)
        print("Closest Musical Note:", closest_note)
        
        return [closest_note, peak_freq]


#### TESTS FOR SINGLE NOTES FILES

Detected_Note1 = note_detect('A3_45.wav')
print("Given: A3")
print("Midi Note = " + str(librosa.note_to_midi(Detected_Note1[0])) + "\n")

Detected_Note2 = note_detect('F4.wav')
print("Given: F4")
print("Midi Note = " + str(librosa.note_to_midi(Detected_Note2[0])) + "\n")

Detected_Note3 = note_detect('A5.wav')
print("Given: A5")
print("Midi Note = " + str(librosa.note_to_midi(Detected_Note3[0])) + "\n")

Detected_Note4 = note_detect('G#5.wav')
print("Given: G#5")
print("Midi Note = " + str(librosa.note_to_midi(Detected_Note4[0])) + "\n")

Detected_Note5 = note_detect('E4.wav')
print("Given: E4")
print("Midi Note = " + str(librosa.note_to_midi(Detected_Note5[0])) + "\n")


############### VISUALIZATION OF MIDI FILE WITH GUI #####################


def visualize_piano_roll(midi_path):
    """Visualizes the MIDI file as a piano roll."""
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 6))
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch
            ax.add_patch(plt.Rectangle((start, pitch - 0.5), end - start, 1, color='blue', alpha=0.7))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MIDI Note Number")
    ax.set_title("Piano Roll Visualization")
    ax.set_xlim(0, midi_data.get_end_time())
    ax.set_ylim(0, 128)
    ax.set_yticks(range(0, 128, 10))
    ax.grid(True)
    return fig

def process_files():
    """Processes the audio and MIDI paths."""
    global midi_path  # To allow access for playback
    audio_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not audio_path:
        messagebox.showerror("Error", "No audio file selected!")
        return

    midi_path = filedialog.asksaveasfilename(defaultextension=".mid", filetypes=[("MIDI files", "*.mid")])
    if not midi_path:
        messagebox.showerror("Error", "No MIDI file name provided!")
        return

    # Convert to MIDI
    audio_to_midi_orig(audio_path, midi_path)

    # Visualize MIDI
    fig = visualize_piano_roll(midi_path)
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Enable Play Button
    play_button.config(state=tk.NORMAL)

def play_midi():
    """Plays the MIDI file."""
    if midi_path:
        pygame.mixer.music.load(midi_path)
        pygame.mixer.music.play()
    else:
        messagebox.showerror("Error", "No MIDI file to play!")

# Initialize the GUI
window = tk.Tk()
window.title("WAV to MIDI Converter and Visualizer")
window.geometry("800x600")

# Initialize Pygame for MIDI playback
pygame.init()
pygame.mixer.init()

# Button for processing files
process_button = tk.Button(window, text="Select WAV File and Specify MIDI File Name", command=process_files, bg="lightblue", font=("Helvetica", 14), relief="solid", padx=20, pady=10)
process_button.pack(pady=20)

# Button for playing MIDI
play_button = tk.Button(window, text="Play MIDI", command=play_midi, bg="lightgreen", font=("Helvetica", 14), relief="solid", padx=20, pady=10)
play_button.pack(pady=10)
play_button.config(state=tk.DISABLED)

window.mainloop()
