from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import soundfile as sf
import tempfile
import os

app = FastAPI()

# Liberar CORS para qualquer origem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def hz_to_note(hz):
    if hz == 0 or np.isnan(hz):
        return None
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if hz < 20:
        return None
    midi = int(np.round(69 + 12 * np.log2(hz / 440.0)))
    note = note_names[midi % 12]
    octave = midi // 12 - 1
    return f"{note}{octave}"

@app.post("/extract_notes/")
async def extract_notes(file: UploadFile = File(...)):
    # Salvar arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=None)
        # Extrair pitches usando librosa.pyin (monofônico)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        notes = []
        for hz in f0:
            note = hz_to_note(hz) if hz is not None else None
            if note:
                notes.append(note)
        # Remover repetições consecutivas
        filtered_notes = []
        for n in notes:
            if not filtered_notes or n != filtered_notes[-1]:
                filtered_notes.append(n)
        return {"notes": filtered_notes}
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.remove(tmp_path)
