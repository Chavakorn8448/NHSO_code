import os
import librosa
import numpy as np
import pandas as pd
from joblib import load
from pathlib import Path
from tqdm import tqdm

MODEL_PATH = "/home/ckancha/rnd/tone_analysis/final-week/deployment/multitask_acoustic_model.joblib"  

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        features = {'file_id': Path(file_path).stem}
    
        # willing_to_serve, not_sluggish, not_monotone
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[pitches > 0]
        if len(pitches) > 0:
            features.update({
                'pitch_mean': np.mean(pitches),
                'pitch_std': np.std(pitches),
                'pitch_jitter': np.mean(np.abs(np.diff(pitches))) / np.mean(pitches) if len(pitches) > 1 else 0
            })
        else:
            features.update({'pitch_mean': 0, 'pitch_std': 0, 'pitch_jitter': 0})
        
        # willing_to_serve, not_sluggish, not_monotone
        energy = librosa.feature.rms(y=y)
        features.update({
            'energy_mean': np.mean(energy),
            'energy_std': np.std(energy)
        })
        
        # not_monotone, not_harsh
        spectral = {
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            'spectral_flatness': np.mean(librosa.feature.spectral_flatness(y=y))
        }
        features.update(spectral)
        
        # not_harsh
        harmonic = librosa.effects.harmonic(y)
        features.update({
            'harmonicity': np.mean(harmonic),
            'hnr': 10 if np.mean(harmonic) > 0.9 else (5 if np.mean(harmonic) > 0.5 else 1)
        })
        
        # willing_to_serve, not_sluggish features
        features['speech_rate'] = len(y) / librosa.get_duration(y=y, sr=sr)
        
        # not_monotone
        features['trailing_slope'] = (energy[0][-1] - energy[0][0]) / len(energy[0]) if len(energy[0]) > 0 else 0
        
        # MFCCs (extra for model)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
        features.update({
            'mfcc1': np.mean(mfccs[0]),
            'mfcc2': np.mean(mfccs[1])
        })
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def load_multitask_model():
    """Load the trained multitask model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return load(MODEL_PATH)

def predict_with_multitask(model, feature_df, thresholds):
    """Run predictions using the multitask model"""
    # Get required features from the model
    required_features = model.feature_names_in_
    
    # Check for missing features
    missing = [f for f in required_features if f not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    # Predict probabilities for all criteria at once
    proba = model.predict_proba(feature_df[required_features])
    
    # Convert to DataFrame with proper column names
    predictions = feature_df.copy()
    for i, criterion in enumerate(thresholds.keys()):
        predictions[f"{criterion}_probability"] = proba[i][:, 1]
        # Add pass/fail column for each criterion
        predictions[f"{criterion}_pass"] = predictions[f"{criterion}_probability"] >= thresholds[criterion]
    
    return predictions

def calculate_criteria_scores(predictions, thresholds):
    """Calculate detailed criteria scores with pass rate information"""
    total_files = len(predictions)
    num_criteria_passed = 0
    criteria_details = {}
    
    for criterion in thresholds:
        # Count files that pass this criterion
        pass_column = f"{criterion}_pass"
        passed_files = predictions[pass_column].sum()
        
        # Calculate pass rate (0.0 to 1.0)
        pass_rate = passed_files / total_files if total_files > 0 else 0
        
        # Score is 1 if pass rate is >= 60%, otherwise 0
        score = 1 if pass_rate >= 0.60 else 0
        
        # Track if this criterion passed the threshold
        if score == 1:
            num_criteria_passed += 1
        
        # result details - optional
        criteria_details[criterion] = {
            'passed_files': int(passed_files),
            'total_files': total_files,
            'pass_rate': pass_rate,
            'score': score
        }
    
    return num_criteria_passed, criteria_details

def tone_analysis(input_dir, twang_score=0):
    # sample threshold
    thresholds = {
        'willing_to_serve': 0.65,
        'not_sluggish': 0.6,
        'not_monotone': 0.7,
        'not_harsh': 0.55
    }
    total_criteria = 5 # plus twang
    
    # input = path?
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return 0, len(thresholds), False
    
    print(f"Starting audio analysis pipeline on directory: {input_dir}")
    
    # extract features
    print("\nExtracting features from audio files...")
    features = []
    audio_files = [f for f in input_dir.glob("*.*") 
                   if f.suffix.lower() in ('.wav', '.mp3', '.flac')]
    
    if not audio_files:
        print(f"Error: No valid audio files found in input directory: {input_dir}")
        return 0, len(thresholds), False
    
    for audio_file in tqdm(audio_files):
        feat = extract_features(audio_file)
        if feat is not None:
            features.append(feat)
    
    if not features:
        print("Error: Failed to extract features from any audio files")
        return 0, len(thresholds), False
    
    feature_df = pd.DataFrame(features)
    
    # load model
    print("\nLoading multitask model...")
    try:
        model = load_multitask_model()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return 0, len(thresholds), False
    
    # predict
    print("\nRunning predictions...")
    try:
        predictions = predict_with_multitask(model, feature_df, thresholds)
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return 0, len(thresholds), False
    
    # calculate criteria scores
    num_criteria_passed, criteria_details = calculate_criteria_scores(predictions, thresholds)
    
    # calculate final result
    total_passed = num_criteria_passed + twang_score
    final_pass = total_passed >= 3

    print("\n=== ANALYSIS RESULTS ===")    
    # result explanation - optional
    for criterion, details in criteria_details.items():
        print(f"{criterion}: {details['passed_files']}/{details['total_files']} files passed ({details['pass_rate']:.2%}) - Score: {details['score']}")
    
    print(f"\nAcoustic criteria passed: {num_criteria_passed}/{len(thresholds)}")
    print(f"Twang score: {twang_score}")
    print(f"Total score: {total_passed}/{total_criteria}")
    print(f"Final Result: {'PASS' if final_pass else 'FAIL'} (4 or more criteria must pass)")
    
    # return num_criteria_passed, total_criteria, final_pass, criteria_details

if __name__ == "__main__":
    input_directory = "/home/ckancha/rnd/tone_analysis/final-week/deployment/demo"
    
    twang_score = 0
    
    tone_analysis(input_directory, twang_score)