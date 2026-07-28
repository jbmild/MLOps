FROM apache/airflow:3.0.2
COPY dockerfiles/airflow/requirements.txt /requirements.txt
COPY src /opt/airflow/src

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         git \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow
ENV PYTHONPATH=/opt/airflow/src

RUN pip install --upgrade pip
RUN pip install --no-cache-dir apache-airflow[virtualenv]==3.0.2 -r /requirements.txt