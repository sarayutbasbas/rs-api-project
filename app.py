from flask import Flask, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json

app = Flask(__name__)
CORS(app)

attributes = {
    "south": [
        'Night',
        'Budget_Transportation',
        'Month',
        'Stay',
        'Who',
        'Salary',
        'Education',
        'Budget_Stay',
        'Activity',
        'Budget_Food',
        'Career',
        'Age_Range',
        'Status',
        'Gender',
        'How_to_go',
        'Religion',
        'Overnight',
        'StreetFood',
        'Package'
    ],
    "north": [
        'Budget_Stay',
        'Age_Range',
        'Activity',
        'Education',
        'Stay',
        'Month',
        'Status',
        'Budget_Food',
        'Budget_Transportation',
    ],
    "central": [
        'Activity',
        'Budget_Food',
        'Career',
        'Month',
        'Age_Range',
        'Stay',
        'Package',
        'Salary'
    ],
    "northeast": [
        'Education',
        'Stay',
        'Budget_Transportation',
        'Budget_Food',
        'Activity',
        'Status',
        'Who',
        'Budget_Stay'
    ],
    "east": [
        'Night',
        'Budget_Transportation',
        'Month',
        'Stay',
        'Who',
        'Salary',
        'Education',
        'Budget_Stay',
        'Activity',
        'Budget_Food',
        'Career',
        'Age_Range',
        'Status',
        'Gender',
        'How_to_go',
        'Religion',
        'Overnight',
        'StreetFood',
        'Package'
    ]
}

def generate_dataframe(data, attributes):
    # สร้าง DataFrame ว่างๆ ก่อน
    print (data, '-->data')
    df = pd.DataFrame()

    # วนลูปเพื่อตรวจสอบและสร้างข้อมูลสำหรับแต่ละ key ที่มีใน attributes
    for key, info in data.items():
        # print(key, info, '-->key, info')
        if key in attributes:
            # สร้าง list ของเลข 0 ด้วยความยาวที่กำหนด
            zeros = [0] * info['count']
            # กำหนดค่า 1 ที่ตำแหน่งที่กำหนด
            zeros[info['value'] - 1] = 1
            # เพิ่ม column ใหม่ใน DataFrame
            for i in range(1, info['count'] + 1):
                df[f'{key}_{i}'] = [zeros[i - 1]]
    
    return df



@app.route('/', methods=['POST'])
def getTop5Location():
    data = request.json
    new_data = generate_dataframe(data['data'], attributes[data['region']])
    print (new_data, '-->new_data')
    new_data.to_csv('na.csv', index=False)

    model_path = {
        'central': 'models/model_Central.pkl',
        'north': 'models/model_North.pkl',
        'south': 'models/model_South.pkl',
        'northeast': 'models/model_Northeast.pkl',
        'east': 'models/model_East.pkl',
    }
    
    model = joblib.load(model_path[data['region']])
    
    # new_data = pd.read_csv('data/dummy-Central-Test.csv')
    probabilities = model.predict_proba(new_data)
    
    top_5_probabilities_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :5]
    top_5_predictions = model.classes_[top_5_probabilities_indices]
    
    print("Top 5 Predicted Classes:")
    data = np.array(top_5_predictions, dtype=object)
    data_list = data.tolist()
    # json_data = json.dumps(data_list)
    # print("NumPy version:", np.__version__)

    return {
        "results": data_list[0]
    }

if __name__ == '__main__':
    app.run(debug=False)