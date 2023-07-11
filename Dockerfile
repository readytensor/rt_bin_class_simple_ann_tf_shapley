FROM python:3.9.17-slim-bullseye as builder


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         ca-certificates \
         dos2unix \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements/requirements.txt /opt/
RUN pip3 install -r /opt/requirements.txt 

COPY src ./opt/src

COPY ./entry_point.sh /opt/
RUN chmod +x /opt/entry_point.sh

COPY ./fix_line_endings.sh /opt/
RUN chmod +x /opt/fix_line_endings.sh 

RUN /opt/fix_line_endings.sh "/opt/src"
RUN /opt/fix_line_endings.sh "/opt/entry_point.sh"

WORKDIR /opt/src

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/src:${PATH}"

USER 1000

ENTRYPOINT ["/opt/entry_point.sh"]