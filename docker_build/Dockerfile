FROM ubuntu:14.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
	g++ \
	gcc \
	git \
	pkg-config \
	wget 

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda2-4.0.5-Linux-x86_64.sh && \
    /bin/bash /Miniconda2-4.0.5-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda2-4.0.5-Linux-x86_64.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda install numpy scipy libpng freetype

#clone bleeding edge matplotlib branch that can properly plot spherical polygons
RUN git clone https://github.com/niknow/matplotlib.git --branch spherical-polygon --single-branch
RUN cd /matplotlib/ && python setup.py build && python setup.py install

#CMD [ "/bin/bash" ]
