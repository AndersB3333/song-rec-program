import pickle
import numpy as np
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify
import pandas as pd


app = Flask(__name__)
CORS(app)
df = pd.read_csv("tracks_eng_language.csv")

popScale = pickle.load(open('musRecPopScale.pkl','rb'))
feaScale = pickle.load(open('musRecFeaScale.pkl', 'rb'))
knn = pickle.load(open("musRecMod.pkl", 'rb'))

@app.route('/', methods=['POST'])
@cross_origin()

def post():
    data = request.get_json()
    song_df = pd.DataFrame.from_dict(data, orient="index").T.rename(columns={"explicit-input": "explicit"})
    print(df.columns)
    num_songs = song_df.iloc[:, -1]
    song_df = song_df.iloc[:, 2:-1]
    song_df = song_df[df.iloc[:, np.r_[1:3, 5:18]].columns]
    scaled_song_rec = feaScale.transform(song_df.iloc[:, 1:])
    rec_popularity = popScale.transform(np.array(song_df.popularity).reshape(1, -1))
    scaled_song_vals = np.concatenate([rec_popularity, scaled_song_rec], axis=1)

    _, indeces = knn.kneighbors(scaled_song_vals)
    knn_index = indeces[0][:int(num_songs.values)]
    recommended_songs = df.iloc[df.index[knn_index].to_list(), [0, 3]].values

    dicty = dict()
    for count in range(len(recommended_songs)):
        dicty[recommended_songs[count][0]] = recommended_songs[count][1]

    response = jsonify(dicty)
    return response

if __name__ == '__main__':
    app.run(port=5000, debug=True)