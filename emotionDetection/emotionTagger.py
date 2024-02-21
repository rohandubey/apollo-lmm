import logging
import essentia
from essentia.standard import MonoLoader, TensorflowPredictVGGish, TensorflowPredict2D
import essentia.streaming as ess
import essentia.standard as es
from essentia.standard import ChordsDetection
import numpy as np
import os
import openai
import ast
import pandas as pd
from tqdm import tqdm
from IPython.display import clear_output
from pydub import AudioSegment

# Set Essentia logging level to suppress warning messages
essentia.log.warningActive = False
logging.getLogger("essentia").setLevel(logging.ERROR)

def process_audio(filename):
    sampleRate = get_sample_rate(filename)
    # Load and process audio for output1
    audio = MonoLoader(filename=filename, sampleRate=sampleRate, resampleQuality=4)()
    predictions = TensorflowPredict2D(graphFilename="emomusic-audioset-vggish-2.pb", output="model/Identity")(
        TensorflowPredictVGGish(graphFilename="audioset-vggish-3.pb", output="model/vggish/embeddings")(audio)
    )
    output1 = [x - 4.5 for x in list(np.mean(predictions, axis=0))]

    # Load and process audio for output2 and global_bpm
    # Initialize algorithms we will use.
    loader = ess.MonoLoader(filename=filename)
    framecutter = ess.FrameCutter(frameSize=4096, hopSize=2048, silentFrames='noise')
    windowing = ess.Windowing(type='blackmanharris62')
    spectrum = ess.Spectrum()
    spectralpeaks = ess.SpectralPeaks(orderBy='magnitude',
                                    magnitudeThreshold=0.00001,
                                    minFrequency=20,
                                    maxFrequency=3500, 
                                    maxPeaks=60)

    hpcp = ess.HPCP()

    # Use pool to store data.
    pool = essentia.Pool() 

    # Connect streaming algorithms.
    loader.audio >> framecutter.signal
    framecutter.frame >> windowing.frame >> spectrum.frame
    spectrum.spectrum >> spectralpeaks.spectrum
    spectralpeaks.magnitudes >> hpcp.magnitudes
    spectralpeaks.frequencies >> hpcp.frequencies
    hpcp.hpcp >> (pool, 'tonal.hpcp')

    # Run streaming network.
    essentia.run(loader)

    chords, strength = ChordsDetection(hopSize=2048, windowSize=2)(pool['tonal.hpcp'])

    audio_11khz = es.MonoLoader(filename=filename, sampleRate=sampleRate)()
    global_bpm, _, _ = es.TempoCNN(graphFilename='deeptemp-k16-3.pb')(audio_11khz)

    # create an audio loader and import audio file for pitch std dev extraction
    sample_rate = sampleRate
    audio = es.MonoLoader(filename=filename, sampleRate=sample_rate)()
    # PitchMelodia takes the entire audio signal as input - no frame-wise processing is required here...
    pExt = es.PredominantPitchMelodia(frameSize=2048, hopSize=256)
    pitch, pitchConf = pExt(audio)

    return output1, chords, global_bpm, np.std(pitch)

def emotion_detection(prompt_string):
  openai.api_type = "azure"
  openai.api_base = "https://aoai-zee5-sb-sc1-musicstudio-0001.openai.azure.com/"
  openai.api_version = "2023-07-01-preview"
  openai.api_key = "4072d7aa8afc4848b8c0a3e4f6e9209a"

  message_text = [{"role":"system","content":"I am giving you a logic for chord to emotion mapping:\nKeys\tEmotion characteristics\nC major\tA pure, certain and decisive manner, full of innocence, earnestness, deepest religious feeling.\nC minor\tExpressive of softness, longing, sadness. Also of earnestness and a passionate intensity. C minor lends itself most effectively to the portraiture of the supernatural. Soft longing, solemnity and dignified earnestness. \nG major\tSoft longing, solemnity and dignified earnestness. Favourite key of youth, expresses sincerity of faith, quiet love, calm meditation, simple grace, pastoral life and a certain humour and brightness.\nG minor\tSometimes sadness, sometimes quiet and sedate joy, a gentle grace with a slight touch of dreamy melancholy. Occasionally it rises to a romantic elevation. It effectively portrays the sentimental; and when used for expressing passionate feelings, the sweetness of its character will deprive passion of all harshness.\nD major\tMajesty, grandeur,and pomp,and adapts itself well to triumphal processions,festival marches,and pictures whichin situations is the prevailing feature\nD minor\tExpresses a subdued feeling of melancholy,grief anxiety,and solemnity\nA major\tFull of confidence and hope,radiant with love and redolentof simple genuine cheerfulness excels all other keys in portraying airy scenesof youth Almost every composerof note has breathed his sincerestand sweetest thoughts into this favourite key\nA minor\tExpressive oftender womanly feeling Most effective for expressingthe quiet melancholy sentimentof Northern nations A minor also expresses sentiments oftender devotion mingled\nB major\tSeldom used. Expresses in fortissimo boldness and pride, pianissimo purity and the most perfect clearness.\nB minor\tVery melancholy, tells of a quiet expectation than yet hopes. It has been observed that nervous persons will sooner be affected by that key than by any other.\nF sharp major\tConquest, relief, triumph, victory, and clarity.\nG flat major\tExpresses softness and coupled with richness.\nF sharp minor\tDark, mysterious, spectral key. Full of passion.\nC sharp major\tScarcely ever used.\nD flat major\tRemarkable for its fullness of tone, and its sonorousness and euphony. It is the favorite key for Notturnos.\nA flat major\tFull of feeling, and replete with dreamy expression.\nA flat minor\tFuneral marches. Full of sad, almost heart rending expression. Wailing of an oppressed, sorrowing heart.\nE flat major\tBoasts the greatest variety of expression. At once serious and solemn, it is the exponent of courage and determination. It gives a piece a brilliant, firm, dignified character. It may be designated as the eminently masculine key.\nE flat minor\tThe darkest and most sombre of all. Rarely used.\nB flat major\tThe favourite of our classical composers. Open, frank, clear, bright. Also an expression of quiet contemplation.\nB flat minor\tFull of gloomy, dark feeling, but seldom used.\nF major\tFull of peace, also expresses effectively a light passing regret or a mournful note of deeply sorrowful feeling. Available for expression of religious sentiment.\nF minor\tA harrowing, full of melancholy. At times rising into passion.\n\nnow for a following dataset having chord, BPM, and pitch variation of music values assign appropriate emotions. Based on diff chords, BPM, and  pitch variation values emotions will be different.\nChords\tBPM\tPitch Variation\nA, F#m, E, Abm, C#m, C#, Am\t126\t236.81903\nBb, A, B, E, Abm, D, Ebm, Gm, C, F, Dm, Eb, G, Em, Am, Bm\t90\t134.17697\nBb, Bbm, Ab, F#, C#, F, Eb, Fm\t30\t136.154\nBb, Bbm, F#, C#, Fm, Am\t30\t139.28336\nBbm, Cm, C#m, Gm, C#, Fm, Em, Am\t84\t398.47635\nBbm, Cm, Gm, C#, G, Em, Am\t84\t415.2539\nA, F#m, E, D, C#m, Em, Bm\t65\t54.151222\nB, Bbm, F#m, E, Cm, D, Ebm, F#, C#m, Ab, C#, C, Dm, Eb, Fm\t96\t99.07164\nE, Cm, D, C, G, Em, Am, Bm\t87\t172.17758\nBb, A, Bbm, Cm, Am, Ebm, Gm, C, F, Dm, Eb, G, Em, Fm\t30\t108.50721\nConsider predicted emotions while taking into account the changes in BPM, and  pitch variation values. A higher pitch can often convey a sense of excitement or elation, while a lower pitch can suggest sadness or seriousness. A lower BPM can induce relaxed state and higher one doesn't. If the chords are same for 2 data, for example C Major but BPM and pitch variations are different then predict the emotions will be guided by chords with addition of few emotion words which are not vague using the BPM and pitch variations. Give me a list with just Predicted Emotion."},{"role":"user","content":"Chords\tBPM\tPitch Variation\nD, Gm, F, C, Bm, Dm, C#, G, Fm, Am, Bb, Cm\t30\t192.09685\nD, Eb, Gm, F, Fm, C, Bm, Ebm, F#m, Em, Dm, A, G, Bbm, Am, Bb, Cm, E\t30\t188.1717\nD, Eb, Gm, F#, Abm, C, Ebm, F#m, C#m, Dm, A, C#, Ab, Fm, Am, Bb, Bbm, E\t97\t164.54123\nEb, Gm, F, C, F#m, A, G, Fm, Am, Bb, Cm, Bbm, Abm\t92\t146.95642\nEb, F#, C#m, C#, Ab, Fm, Cm, Abm\t67\t152.53105\nD, Eb, Gm, F, F#, Fm, Abm, Bm, Ebm, F#m, Dm, C#, Ab, G, Bbm, Bb, Cm, E\t30\t229.39914\nD, F, Abm, C, Ebm, F#m, C#m, Em, Dm, B, A, C#, Fm, Am, Bm, E\t30\t243.95262\nD, C#m, A, C#, Abm, Gm, C, F#m, Ab, Cm, Eb, F#, Em, G, E, Ebm, Dm, B, Fm, Bm\t30\t96.604416\nC#m, A, C#, Abm, Ab, Am, Bbm, Cm, Eb, F, F#, Em, E, Bm, Ebm, Dm, B, Fm, Bb\t30\t162.93541\nEb, D, F#, F#m, Em, Dm, B, A, Ab, G, Am, Bm, E, Abm\t30\t171.14784\nD, F, Bb, F#m, C#m, Dm, B, A, Ab, G, Am, Bm, Cm, E\t90\t154.01668"},{"role":"assistant","content":"Predicted Emotion\nSerenity, Devotion, Peaceful Contemplation\nDevotion, Love, Compassion, Yearning\nGrandeur, Joy, Majesty, Devotion\nMelancholy, Yearning, Devotion, Tranquility\nJoy, Playfulness, Celebration\nPathos, Introspection, Seriousness\nMorning Serenity, Devotion, Tranquility\nMelancholy, Grandeur, Deep Emotions, Devotion\nContemplation, Serenity, Calmness, Mystical\nmood of joy, devotion, and serenity\nsadness, romance, peace strength, courage, dispassion"}]
  new_user_message = {"role": "user", "content": prompt_string.lower()}
  message_text.append(new_user_message)

  completion = openai.ChatCompletion.create(
    engine="test-gpt4",
    messages = message_text,
    temperature=0.7,
    max_tokens=400,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
  )
  return completion['choices'][0]['message']['content']

def get_sample_rate(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        sample_rate = audio.frame_rate
        return sample_rate
    except Exception as e:
        print(f"Error: {e}")
        return None

path_to_check = "ZeeRishtey"

# List of files to check
files_to_check = [file for file in os.listdir(path_to_check) if file.lower().endswith((".wav", ".mp3"))]
data = []

# Use tqdm.notebook for Jupyter notebook compatibility
for file_to_check in tqdm(files_to_check, desc="Processing Files"):
    file_path = os.path.join(path_to_check, file_to_check)
    # output1, chords, global_bpm, pitch_std = process_audio(file_path)  # Assuming 44100 is the sample rate
    global_bpm= process_audio(file_path)  # Assuming 44100 is the sample rate

    data.append([file_to_check, global_bpm])
    # data.append([file_to_check, output1, str(list(set(chords))).replace('\'', '').replace('[', '').replace(']', ''), global_bpm, pitch_std])
    
    # Clear the output to update the progress bar without printing new lines
    clear_output(wait=True)

result_list = [(param, list(set(inner_list[1])), inner_list[2], inner_list[3]) for param, inner_list in data]
# Define the columns
columns = ['file_name', 'NotesProgression', 'BPM', 'PitchVariation']

# Create a DataFrame
df = pd.DataFrame(result_list, columns=columns)
df['Emotion'] = [line.strip() for line in emotion_detection(str(df.loc[:, ['NotesProgression', 'BPM', 'PitchVariation']].values.flatten().tolist())).split('\n')][1:]

# Iterate through the DataFrame and create text files
for index, row in df.iterrows():
    file_name = row['file_name']
    emotion = row['Emotion']

    # Extract file extension
    file_extension = os.path.splitext(file_name)[1]

    # Create the file path
    file_path = os.path.join(path_to_check, file_name.replace(file_extension, '.txt'))

    # Write the emotion to the text file
    with open(file_path, 'w') as file:
        file.write(emotion)

# Write DataFrame to CSV file
output_path = os.path.join(path_to_check, 'acguitar_data.csv')
df.to_csv(output_path, index=False)