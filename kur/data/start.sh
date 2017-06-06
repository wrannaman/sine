python3 generate-data.pickle.py 100000 train.pkl
python3 generate-data.pickle.py 10000 validate.pkl
python3 generate-data.pickle.py 10000 test.pkl
python3 generate-data.pickle.py 10000 evaluate.pkl

python3 inspect-data.py
python3 chart-data.py
