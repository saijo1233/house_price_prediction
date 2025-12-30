FROM jupyter/minimal-notebook:latest

USER root

RUN pip install numpy pandas lightgbm scikit-learn matplotlib psycopg2-binary sqlalchemy

USER jovyan