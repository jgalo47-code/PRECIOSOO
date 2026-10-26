# Campus Facility Reservation Monitoring (Streamlit)
Reuses the midterm Decision Tree (target `Reservation_Outcome`: Completed / Not Completed). Decision support only.
```
pip install -r requirements.txt
# put your cleaned midterm CSV at data/Facility_reservation_cleaned.csv
python train_model.py && python make_diagram.py
streamlit run app.py
```
Deploy via share.streamlit.io (entry point `app.py`). Pin the scikit-learn version used for training. Use synthetic/authorized data only.
