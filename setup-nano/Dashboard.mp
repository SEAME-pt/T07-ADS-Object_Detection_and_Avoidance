sudo apt install python3-dev 

pip3 install --upgrade pip setuptools packaging

python3 -m venv env_dash

source env_dash/bin/activate 

pip install -r requirements.txt

to run:

source env_dash/bin/activate

python3 Backend/app.py