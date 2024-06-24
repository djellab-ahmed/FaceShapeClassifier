from flask import Flask, request, render_template, redirect, url_for, flash
import cv2
import imutils
import os
from werkzeug.utils import secure_filename
import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np
import pandas as pd
from joblib import load


filter_path = "models_last/shape_predictor_81_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(filter_path)

distances_map = {
    'D1': (75, 76, 79),
    'D2': (0, 16),
    'D3': (69, 72, 8),
    
    # D4 = ( D4 + D5 + D6 + D7 + D8 ) * 2 jawline width 
    'D4': (8, 9),
    'D5': (9, 10),
    'D6': (10, 11),
    'D7': (11, 12),
    'D8': (12, 13),
    
    'D9': (69, 27),
    'D10': (27, 30),
    'D11': (30, 8),

    'D12': (2, 14),  
    'D13': (4, 12),  
    'D14': (6, 10),
    'D15': (7, 9)
    
}

distances_keys = ['D1', 'D2', 'D3']

distances_keys_ver = ['D9', 'D10', 'D11']

landmarks_to_extract = []
for i in range(0, 81):
    landmarks_to_extract.append(i)


def euclidean_distance(pt1, pt2):
    """Compute Euclidean distance between two points."""
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)  

def compute_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def compute_curvature(points):
    # Fit a quadratic curve (parabola) to the points
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])

    # Quadratic coefficient (second-order polynomial)
    coefficients = np.polyfit(x, y, 2)
    a, b, _ = coefficients

    # Compute the curvature from the coefficients of the quadratic curve
    # For simplicity, we calculate curvature at the middle point x value
    x_middle = x[len(x) // 2]
    curvature = np.abs(2 * a) / (1 + (2 * a * x_middle + b) ** 2) ** 1.5
    
    return curvature

def extract_features(image):
    
    combined_df = pd.DataFrame()

    # detect faces in the image
    rects = detector(image, 1)

    for rect in rects:
    
        landmarks = predictor(image, rect)
        landmarks = face_utils.shape_to_np(landmarks)
         
        
        # Extract landmarks
        extracted_landmarks = {i: landmarks[i] for i in landmarks_to_extract if i < len(landmarks) }
    
    
        # Create sets of points for the jawline, forehead, and cheekbone
        jawline_points = [extracted_landmarks[i] for i in range(4, 13)]
        forehead_points = [extracted_landmarks[i] for i in range(69, 72+1)] # Including 72 as the last index
        cheekbone_points = [extracted_landmarks[i] for i in [0, 16, 1, 15, 2, 14, 3, 13]]
    
        chin_points = [extracted_landmarks[i] for i in range(6, 11+1)]
        eyebrow_points = [extracted_landmarks[i] for i in range(17, 26+1)]  # Assuming this range includes both eyebrows
        temple_points = [extracted_landmarks[i] for i in range(70, 75+1)]  # Assuming these are the temple points
        nose_bridge_points = [extracted_landmarks[i] for i in [27, 28, 29]]
        upper_lip_points = [extracted_landmarks[i] for i in range(48, 54+1)]
        hairline_points = [extracted_landmarks[i] for i in range(69, 71+1)]
        
        # Dictionary to save computed distances
        computed_distances = {}
        
        # Compute distances
        for distance_name, landmark_numbers in distances_map.items():
            if len(landmark_numbers) == 2:
                # Standard distance computation for pairs
                pt1 = extracted_landmarks[landmark_numbers[0]]
                pt2 = extracted_landmarks[landmark_numbers[1]]
                distance = euclidean_distance(pt1, pt2)
            elif len(landmark_numbers) == 3:
                # Special case for D3: distance from hairline to the chin
                mid_pt = ((extracted_landmarks[landmark_numbers[0]][0] + extracted_landmarks[landmark_numbers[1]][0]) / 2, 
                          (extracted_landmarks[landmark_numbers[0]][1] + extracted_landmarks[landmark_numbers[1]][1]) / 2)
                distance = euclidean_distance(mid_pt, extracted_landmarks[landmark_numbers[2]])
            else:
                raise ValueError(f"Unexpected number of landmarks for {distance_name}")
        
            # Save to the computed_distances dictionary
            computed_distances[distance_name] = distance


        selected_computed_distances = {key: computed_distances[key] for key in distances_keys}
    
        selected_computed_distances_ver = {key: computed_distances[key] for key in distances_keys_ver}
    
        #selected_computed_distances = {key: computed_distances[key] for key in distances_keys}
        jawline_length = computed_distances['D4'] + computed_distances['D5'] + computed_distances['D6'] + computed_distances['D7'] + computed_distances['D8']
        jawline_length = jawline_length * 2
    
        del computed_distances['D4']
        del computed_distances['D5']
        del computed_distances['D6']
        del computed_distances['D7']
        del computed_distances['D8']
    
        selected_computed_distances['D4'] = jawline_length
        selected_computed_distances['D5'] = computed_distances['D12']
        selected_computed_distances['D6'] = computed_distances['D13']
        selected_computed_distances['D7'] = computed_distances['D14']
        selected_computed_distances['D8'] = computed_distances['D15']
        
        
        computed_distances['D4'] = jawline_length

    
        # Now, normalize the distances according to the provided formula
        total_distance = sum(selected_computed_distances.values())
        normalized_distances = {f'N{index + 1}': (value / total_distance) for index, value in enumerate(selected_computed_distances.values())}
    
        # Now, normalize the distances according to the provided formula
        total_distance = sum(selected_computed_distances_ver.values())
        normalized_distances_ver = {f'N{index + 9}': (value / total_distance) for index, value in enumerate(selected_computed_distances_ver.values())}
    
    
        # Calculate the ratios
        
        ratios = {
            'R1': computed_distances['D1'] / computed_distances['D2'],
            'R2': computed_distances['D1'] / computed_distances['D3'],
            'R3': computed_distances['D1'] / computed_distances['D4'],
            'R4': computed_distances['D2'] / computed_distances['D3'],
            'R5': computed_distances['D2'] / computed_distances['D4'],
            'R6': computed_distances['D3'] / computed_distances['D4'],
    
            'R7': computed_distances['D9'] / computed_distances['D10'],
            'R8': computed_distances['D10'] / computed_distances['D11'],
            'R9': computed_distances['D9'] / computed_distances['D11'],
            
            'R10': computed_distances['D12'] / computed_distances['D3'],
            'R11': computed_distances['D1'] / computed_distances['D3'],
            'R12': computed_distances['D1'] / computed_distances['D12'],
            'R13': computed_distances['D13'] / computed_distances['D1'],
            'R14': computed_distances['D14'] / computed_distances['D12'],
            'R15': computed_distances['D15'] / computed_distances['D14'],
            'R16': computed_distances['D12'] / computed_distances['D13']
                    
        }

    
        # Compute the angle
        A3 = compute_angle(extracted_landmarks[12], extracted_landmarks[14], extracted_landmarks[2])
        
        center_a2 = ( extracted_landmarks[69] + extracted_landmarks[72] ) / 2
        A2 = compute_angle(center_a2, extracted_landmarks[8], extracted_landmarks[12])
        
        A1 = compute_angle(center_a2, extracted_landmarks[8], extracted_landmarks[10])
    
        # Jawline Angle
        A4 = compute_angle(extracted_landmarks[4], extracted_landmarks[8], extracted_landmarks[12])
    
        # Forehead Slope
        A5 = compute_angle(extracted_landmarks[69], extracted_landmarks[19], extracted_landmarks[24])
    
        #selected_normalized_distances = {key: normalized_distances[key] for key in normalized_distances_keys}
        #selected_ratios = {key: ratios[key] for key in ratio_keys}
    
        selected_angles  ={'A1': A1,
                          'A2': A2,
                           'A3': A3,
                           'A4': A4,
                           'A5': A5
                          }
    
    
        # Compute curvatures for each facial region
        curvatures = {
            'jawline_curvature': compute_curvature(jawline_points),
            'forehead_curvature': compute_curvature(forehead_points),
            'cheekbone_curvature': compute_curvature(cheekbone_points),
            'chin_curvature': compute_curvature(chin_points),
            'eyebrow_bone_curvature': compute_curvature(eyebrow_points),
            'temple_curvature': compute_curvature(temple_points),
            'nose_bridge_curvature': compute_curvature(nose_bridge_points),
            'upper_lip_curvature': compute_curvature(upper_lip_points),
            'hairline_curvature': compute_curvature(hairline_points)
        }
    
        # Combine the dictionaries
        combined_dict = {**normalized_distances, **normalized_distances_ver, **ratios, **selected_angles, **curvatures}
        
        # Append the combined dictionary as a new row to the DataFrame
        combined_df = pd.concat([combined_df, pd.DataFrame([combined_dict])], ignore_index=True)
    
    
        # Draw landmarks
        #for landmark_num, (x, y) in extracted_landmarks.items():
            #cv2.circle(image, (x, y), 1, (0, 255, 0), 3)
            #cv2.putText(image, str(landmark_num), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
        for distance_name, landmark_numbers in distances_map.items():
                if len(landmark_numbers) == 2:
                    pt1 = extracted_landmarks[landmark_numbers[0]]
                    pt2 = extracted_landmarks[landmark_numbers[1]]
                    mid_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                    cv2.line(image, pt1, pt2, (0, 255, 255), 1)
                    #if (distance_name != 'D5' and distance_name != 'D6' and distance_name != 'D7'):
                        #cv2.putText(image, distance_name, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                elif len(landmark_numbers) == 3:
                    mid_pt = ((extracted_landmarks[landmark_numbers[0]][0] + extracted_landmarks[landmark_numbers[1]][0]) // 2, 
                              (extracted_landmarks[landmark_numbers[0]][1] + extracted_landmarks[landmark_numbers[1]][1]) // 2)
                    cv2.line(image, mid_pt, extracted_landmarks[landmark_numbers[2]], (0, 255, 255), 1)
                    mid_line_pt = ((mid_pt[0] + extracted_landmarks[landmark_numbers[2]][0]) // 2, 
                                   (mid_pt[1] + extracted_landmarks[landmark_numbers[2]][1]) // 2)
                    #if (distance_name != 'D5' and distance_name != 'D6' and distance_name != 'D7'):
                        #cv2.putText(image, distance_name, mid_line_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    
    return combined_df, extracted_landmarks

# Load the model and scaler
forest_loaded = load('models_last/random_forest_model.joblib')
scaler_loaded = load('models_last/scaler.joblib')

# Mapping of encoded values to class names
class_mapping = {
    0: 'Heart',
    1: 'Oval',
    2: 'Round',
    3: 'Square'
}

UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'static/processed/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Image processing
        image = cv2.imread(file_path)
        image = imutils.resize(image, width=500)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        feature_vector, raw_landmarks = extract_features(image)
        # Convert landmarks to a format suitable for JavaScript
        #landmarks = [{'x': int(point[0]), 'y': int(point[1])} for point in raw_landmarks]
        landmarks_list = [{'x': int(point[0]), 'y': int(point[1])} for point in raw_landmarks.values()]

        X_new_scaled = scaler_loaded.transform(feature_vector)
        predictions = forest_loaded.predict(X_new_scaled)
        prediction_encoded = predictions[0]
        class_name = class_mapping[prediction_encoded]

        # No need to draw landmarks here
        # Save the original image (or a resized version of it)
        processed_filename = 'processed_' + filename
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        cv2.imwrite(processed_path, image)

        return render_template('upload_3.html', landmarks=landmarks_list, image_url=url_for('static', filename='processed/' + processed_filename), class_name=class_name)

    return redirect(url_for('index'))

# Function to run the Flask app
def run_app():
    app.run(host='0.0.0.0', debug=True, port=5010)

if __name__ == "__main__":
    run_app()

